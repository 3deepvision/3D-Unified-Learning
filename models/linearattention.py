import torch
import torch.nn as nn

from einops import rearrange



def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = (
            torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        )

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(
        self, hidden_dim, guidance_dim, nheads=8, attention_type="linear"
    ):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)


        self.attention = LinearAttention()



    def forward(self, x):
        """
        Arguments:
            x: B, L, C
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.nheads)
        k = rearrange(k, "B S (H D) -> B S H D", H=self.nheads)
        v = rearrange(v, "B S (H D) -> B S H D", H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, "B L H D -> B L (H D)")
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        guidance_dim=64,
        nheads=8,
        attention_type="linear",
    ) -> None:
        super().__init__()
        self.attention = AttentionLayer(
            hidden_dim,
            guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.skipmlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),)
        self.skipnorm = nn.LayerNorm(hidden_dim)




    def forward(self, x):
        """
        Arguments:
            x: B, C, T, N
            base_pred: N, 1
        """
        B, _, _, N = x.size()
        x_pool = rearrange(x, "B C T N -> (B N) T C")
        x_residual = x_pool

        # if base_pred is not None:
        #     x_bg = x_pool[:, :1].clone()
        #     x_pool[:, :1] = self.base_merge(
        #         self.norm_xbg(
        #             torch.cat(
        #                 [
        #                     x_bg,
        #                     base_pred.unsqueeze(-1).repeat(
        #                         1, 1, x_bg.shape[-1]
        #                     ),
        #                 ],
        #                 dim=1,
        #             )
        #         )
        #     )  # N, 2, C

        x_pool = x_pool + self.attention(self.norm1(x_pool))
        x_pool = self.skipmlp(self.skipnorm(x_pool)) + self.MLP(self.norm2(x_pool))

        x_pool = rearrange(x_pool, "(B N) T C -> B C T N", N=N)

        # 1x1 残差连接 对其通道
        x_residual = self.skipmlp(self.skipnorm(x_residual))
        x_residual = rearrange(x_residual, "(B N) T C -> B C T N", N=N)



        x = x_residual + x_pool  # Residual
        return x


class SpatialTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        guidance_dim=64,
        nheads=8,
        attention_type="linear",
    ) -> None:
        super().__init__()
        self.attention = AttentionLayer(
            hidden_dim,
            guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Arguments:
            x: B, C, N
        """
        B, C, N = x.size()

        x_pool = rearrange(x, "B C N -> B N C")

        x_pool = x_pool + self.attention(self.norm1(x_pool))  # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool))  # MLP

        x_pool = rearrange(x_pool, "B N C -> B C N")

        x = x + x_pool  # Residual
        return x


class AggregatorLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        guidance_dim=512,
        nheads=4,
        attention_type="linear",
    ) -> None:
        super().__init__()
        self.spatial_attention = SpatialTransformerLayer(
            hidden_dim,
            guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )

        self.class_attention = ClassTransformerLayer(
            hidden_dim,
            guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )

        self.init_weights()


    def init_weights(self):
        for name, m in self.named_parameters():
            if "class_attention.base_merge" in name:
                continue
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)


    def forward(self, x, basept_guidance):
        """
        Arguments:
            x: B C T N   B为batchsize C为点的通道 T为特征图数量 N为点数量
        """
        x = self.spatial_attention(x)
        x = self.class_attention(x, basept_guidance)
        return x


