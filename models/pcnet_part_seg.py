from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from linearattention import ClassTransformerLayer,SpatialTransformerLayer
from assistingdgcnn import Assist_DGCNN
from leader_pct import Leader_Point_Transformer_partseg

class get_model(nn.Module):
    def __init__(self, num_class=3, normal_channel=True, hidden_dim=64, guidance_dim=0, nheads=4):

        super(get_model, self).__init__()

        ###Transformer 参数
        self.hidden_dim = hidden_dim
        self.guidance_dim = guidance_dim
        self.nheads = nheads
        ###

        self.num_class = num_class

        self.assist_dgcnn = Assist_DGCNN()
        self.leader_pct = Leader_Point_Transformer_partseg(self.num_class, self.hidden_dim, self.guidance_dim, self.nheads)



        self.class_attention = ClassTransformerLayer(self.hidden_dim,
                                                         self.guidance_dim,
                                                         nheads=self.nheads)

        self.spatial_attention = SpatialTransformerLayer(self.hidden_dim * 2,
                                                         self.guidance_dim,
                                                         nheads=self.nheads)


        self.skipmlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),)
        self.skipnorm = nn.LayerNorm(hidden_dim)


       # self.init_weights()


    # def init_weights(self):
    #     for name, m in self.named_parameters():
    #         if m.dim() > 1:
    #             nn.init.xavier_uniform_(m)



    def forward(self, x, cls_label):
        batch_size, _, N = x.size()
        x1, x2, x3, x4 = self.assist_dgcnn(x)
        x1 = x1.unsqueeze(2)  # B C N -> B C 1 N
        x2 = x2.unsqueeze(2)
        x3 = x3.unsqueeze(2)
        x4 = x4.unsqueeze(2)


        fusedf1 = torch.cat((x1, x2, x3, x4),dim=2) # B C N -> B C T N
        fusedf2 = torch.cat((x2, x3, x4), dim=2)
        fusedf3 = torch.cat((x3, x4), dim=2)
        fusedf4 = self.skipmlp(self.skipnorm(x4.squeeze(2).permute(0,2,1))).permute(0,2,1)

        fusedf1 = self.class_attention(fusedf1)
        fusedf1 = rearrange(fusedf1, "B C T N -> (B T) N C")
        fusedf1 = self.spatial_attention(fusedf1)
        fusedf1 = rearrange(fusedf1, "(B T) N C -> B C T N", N=N)

        fusedf2 = self.class_attention(fusedf2)
        fusedf2 = rearrange(fusedf2, "B C T N -> (B T) N C")
        fusedf2 = self.spatial_attention(fusedf2)
        fusedf2 = rearrange(fusedf2, "(B T) N C -> B C T N", N=N)

        fusedf3 = self.class_attention(fusedf3)
        fusedf3 = rearrange(fusedf3, "B C T N -> (B T) N C")
        fusedf3 = self.spatial_attention(fusedf3)
        fusedf3 = rearrange(fusedf3, "(B T) N C -> B C T N", N=N)

        fusedf1 = torch.sum(fusedf1, dim=2)
        fusedf2 = torch.sum(fusedf2, dim=2)
        fusedf3 = torch.sum(fusedf3, dim=2)

        # print(fusedf1.shape)
        # print(fusedf2.shape)
        # print(fusedf3.shape)
        # print(fusedf4.shape)


        return self.leader_pct(x, fusedf1, fusedf2, fusedf3, fusedf4, cls_label)



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss



