import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import math
from linearattention import SpatialTransformerLayer


class Leader_Point_Transformer_semseg(nn.Module):
    def __init__(self, num_class=3, hidden_dim=64, guidance_dim=0, nheads=4):
        super(Leader_Point_Transformer_semseg, self).__init__()
        self.num_class = num_class

        self.spatial_attention1 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)
        self.spatial_attention2 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)
        self.spatial_attention3 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)
        self.spatial_attention4 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)



        self.conv1 = nn.Conv1d(9, 128, kernel_size=1, bias=False)  # normal时=6
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.convmsf1 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf2 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf3 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf4 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.bnmsf1 = nn.BatchNorm1d(128)
        self.bnmsf2 = nn.BatchNorm1d(128)
        self.bnmsf3 = nn.BatchNorm1d(128)
        self.bnmsf4 = nn.BatchNorm1d(128)
        self.mu1 = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
        self.mu2 = nn.Parameter(torch.tensor([0.15], dtype=torch.float32))
        self.mu3 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.mu4 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))


        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.2))


        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.num_class, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, f1, f2, f3, f4):    # x = [8,3,4096], cls_label = [8,1]
        batch_size, C, N = x.size()
        # print(x.shape)                # x = [8,6,4096]
        # print(cls_label.shape)        # cls_label = [8,1]
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N   [8,128,4096]
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N   [8,128,4096]

        x_right = torch.cat((x, f1), dim=1)
        x_right = F.relu(self.bnmsf1(self.convmsf1(x_right)))
        x_up = torch.roll(x, -1, 2)
        x_dowm = torch.roll(x, 1, 2)
        f1_up = torch.roll(f1, -1, 2)
        f1_down = torch.roll(f1, 1, 2)
        x_left = torch.cat(
            (x_up[:, :C / 4, :], x_dowm[:, C / 4:C / 2, :], f1_up[:, C / 2:C * 3 / 4, :], f1_down[:, C * 3 / 4:, :]),
            dim=1)
        x = x_left * self.mu1 + x_right
        x = self.spatial_attention1(x)

        x1 = self.sa1(x)  # B, D, N   [8,128,4096]

        x1_right = torch.cat((x1, f2), dim=1)
        x1_right = F.relu(self.bnmsf2(self.convmsf2(x1_right)))
        x1_up = torch.roll(x1, -1, 2)
        x1_dowm = torch.roll(x1, 1, 2)
        f2_up = torch.roll(f2, -1, 2)
        f2_down = torch.roll(f2, 1, 2)
        x1_left = torch.cat(
            (x1_up[:, :C / 4, :], x1_dowm[:, C / 4:C / 2, :], f2_up[:, C / 2:C * 3 / 4, :], f2_down[:, C * 3 / 4:, :]),
            dim=1)
        x1 = x1_left * self.mu2 + x1_right
        x1 = self.spatial_attention2(x1)

        x2 = self.sa2(x1) # B, D, N   [8,128,4096]

        x2_right = torch.cat((x2, f3), dim=1)
        x2_right = F.relu(self.bnmsf3(self.convmsf3(x2_right)))
        x2_up = torch.roll(x2, -1, 2)
        x2_dowm = torch.roll(x2, 1, 2)
        f3_up = torch.roll(f3, -1, 2)
        f3_down = torch.roll(f3, 1, 2)
        x2_left = torch.cat(
            (x2_up[:, :C / 4, :], x2_dowm[:, C / 4:C / 2, :], f3_up[:, C / 2:C * 3 / 4, :], f3_down[:, C * 3 / 4:, :]),
            dim=1)
        x2 = x2_left * self.mu3 + x2_right
        x2 = self.spatial_attention3(x2)

        x3 = self.sa3(x2) # B, D, N   [8,128,4096]

        x3_right = torch.cat((x3, f4), dim=1)
        x3_right = F.relu(self.bnmsf4(self.convmsf4(x3_right)))
        x3_up = torch.roll(x3, -1, 2)
        x3_dowm = torch.roll(x3, 1, 2)
        f4_up = torch.roll(f4, -1, 2)
        f4_down = torch.roll(f4, 1, 2)
        x3_left = torch.cat(
            (x3_up[:, :C / 4, :], x3_dowm[:, C / 4:C / 2, :], f4_up[:, C / 2:C * 3 / 4, :], f4_down[:, C * 3 / 4:, :]),
            dim=1)
        x3 = x3_left * self.mu4 + x3_right
        x3 = self.spatial_attention3(x3)

        x4 = self.sa4(x3) # B, D, N   [8,128,4096]



        if self.training:
            x = torch.cat((x1, x2, x3, x4), dim=1)  # B, D, N   [8,512,4096]
            assist_x = torch.cat((f1, f2, f3, f4), dim=1)  # B, D, N   [8,512,4096]

            x = self.conv_fuse(x)  # B, D, N   [8,1024,4096]
            assist_x = self.conv_fuse(assist_x)  # B, D, N   [8,1024,4096]

            x_max = x.max(dim=2, keepdim=False)[0]  # B, D      [8,1024]
            assist_x_max = assist_x.max(dim=2, keepdim=False)[0]

            x_avg = x.mean(dim=2, keepdim=False)  # B, D      [8,1024]
            assist_x_avg = assist_x.mean(dim=2, keepdim=False)

            x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]
            assist_x_max_feature = assist_x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]

            x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]
            assist_x_avg_feature = assist_x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]

            # cls_label_one_hot = cls_label.view(batch_size, 1, 1)  # [8,1,1]
            # cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # B, D, N   [8,64,4096]

            x_global_feature = torch.cat((x_max_feature, x_avg_feature),
                                         1)  # 1024*2   # B, D, N   [8,2048,4096]
            assist_x_global_feature = torch.cat((assist_x_max_feature, assist_x_avg_feature),
                                         1)  # 1024*2    # B, D, N   [8,2048,4096]

            x = torch.cat((x, x_global_feature), 1)  # 1024 * 3    # B, D, N   [8,3072,4096]
            assist_x = torch.cat((assist_x, assist_x_global_feature), 1)  # 1024 * 3    # B, D, N   [8,3136,4096]

            x = self.relu(self.bns1(self.convs1(x)))  # B, D, N   [8,512,4096]
            assist_x = self.relu(self.bns1(self.convs1(assist_x)))

            x = self.dp1(x)  # B, D, N   [8,512,4096]
            assist_x = self.dp1(assist_x)  # B, D, N   [8,512,4096]

            x = self.relu(self.bns2(self.convs2(x)))  # B, D, N   [8,256,4096]
            assist_x = self.relu(self.bns2(self.convs2(assist_x)))  # B, D, N   [8,256,4096]

            x = self.convs3(x)  # B, D, N   [8,2,4096]
            assist_x = self.convs3(assist_x)

            x = x.permute(0, 2, 1)
            assist_x = assist_x.permute(0, 2, 1)


            return x, assist_x


        else:

            x = torch.cat((x1, x2, x3, x4), dim=1)    # B, D, N   [8,512,4096]
            x = self.conv_fuse(x)                     # B, D, N   [8,1024,4096]
            x_max = x.max(dim=2, keepdim=False)[0]    # B, D      [8,1024]

            x_avg = x.mean(dim=2, keepdim=False)      # B, D      [8,1024]
            x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)    # B, D, N   [8,1024,4096]
            x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)    # B, D, N   [8,1024,4096]

            x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1) # 1024*2   # B, D, N   [8,2048,4096]
            x = torch.cat((x, x_global_feature), 1) # 1024 * 3    # B, D, N   [8,3072,4096]
            x = self.relu(self.bns1(self.convs1(x)))                  # B, D, N   [8,512,4096]
            x = self.dp1(x)                                           # B, D, N   [8,512,4096]
            x = self.relu(self.bns2(self.convs2(x)))                  # B, D, N   [8,256,4096]
            x = self.convs3(x)                                        # B, D, N   [8,2,4096]

            x = x.permute(0, 2, 1)

            return x, None







class Leader_Point_Transformer_partseg(nn.Module):
    def __init__(self, num_class=3, hidden_dim=64, guidance_dim=0, nheads=4):
        super(Leader_Point_Transformer_partseg, self).__init__()
        self.num_class = num_class

        self.spatial_attention1 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)
        self.spatial_attention2 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)
        self.spatial_attention3 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)
        self.spatial_attention4 = SpatialTransformerLayer(hidden_dim * 2,
                                                          guidance_dim,
                                                          nheads=nheads)



        self.conv1 = nn.Conv1d(6, 128, kernel_size=1, bias=False)  # normal时=6
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.convmsf1 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf2 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf3 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf4 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.bnmsf1 = nn.BatchNorm1d(128)
        self.bnmsf2 = nn.BatchNorm1d(128)
        self.bnmsf3 = nn.BatchNorm1d(128)
        self.bnmsf4 = nn.BatchNorm1d(128)
        self.mu1 = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
        self.mu2 = nn.Parameter(torch.tensor([0.15], dtype=torch.float32))
        self.mu3 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.mu4 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(1, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.num_class, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, f1, f2, f3, f4, cls_label):    # x = [8,3,4096], cls_label = [8,1]
        batch_size, C, N = x.size()
        # print(x.shape)                # x = [8,6,4096]
        # print(cls_label.shape)        # cls_label = [8,1]
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N   [8,128,4096]
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N   [8,128,4096]

        x_right = torch.cat((x, f1), dim=1)
        x_right = F.relu(self.bnmsf1(self.convmsf1(x_right)))
        x_up = torch.roll(x, -1, 2)
        x_dowm = torch.roll(x, 1, 2)
        f1_up = torch.roll(f1, -1, 2)
        f1_down = torch.roll(f1, 1, 2)
        x_left = torch.cat(
            (x_up[:, :C / 4, :], x_dowm[:, C / 4:C / 2, :], f1_up[:, C / 2:C * 3 / 4, :], f1_down[:, C * 3 / 4:, :]),
            dim=1)
        x = x_left * self.mu1 + x_right
        x = self.spatial_attention1(x)


        x1 = self.sa1(x)  # B, D, N   [8,128,4096]

        x1_right = torch.cat((x1, f2), dim=1)
        x1_right = F.relu(self.bnmsf2(self.convmsf2(x1_right)))
        x1_up = torch.roll(x1, -1, 2)
        x1_dowm = torch.roll(x1, 1, 2)
        f2_up = torch.roll(f2, -1, 2)
        f2_down = torch.roll(f2, 1, 2)
        x1_left = torch.cat(
            (x1_up[:, :C / 4, :], x1_dowm[:, C / 4:C / 2, :], f2_up[:, C / 2:C * 3 / 4, :], f2_down[:, C * 3 / 4:, :]),
            dim=1)
        x1 = x1_left * self.mu2 + x1_right
        x1 = self.spatial_attention2(x1)

        x2 = self.sa2(x1) # B, D, N   [8,128,4096]

        x2_right = torch.cat((x2, f3), dim=1)
        x2_right = F.relu(self.bnmsf3(self.convmsf3(x2_right)))
        x2_up = torch.roll(x2, -1, 2)
        x2_dowm = torch.roll(x2, 1, 2)
        f3_up = torch.roll(f3, -1, 2)
        f3_down = torch.roll(f3, 1, 2)
        x2_left = torch.cat(
            (x2_up[:, :C / 4, :], x2_dowm[:, C / 4:C / 2, :], f3_up[:, C / 2:C * 3 / 4, :], f3_down[:, C * 3 / 4:, :]),
            dim=1)
        x2 = x2_left * self.mu3 + x2_right
        x2 = self.spatial_attention3(x2)

        x3 = self.sa3(x2) # B, D, N   [8,128,4096]

        x3_right = torch.cat((x3, f4), dim=1)
        x3_right = F.relu(self.bnmsf4(self.convmsf4(x3_right)))
        x3_up = torch.roll(x3, -1, 2)
        x3_dowm = torch.roll(x3, 1, 2)
        f4_up = torch.roll(f4, -1, 2)
        f4_down = torch.roll(f4, 1, 2)
        x3_left = torch.cat(
            (x3_up[:, :C / 4, :], x3_dowm[:, C / 4:C / 2, :], f4_up[:, C / 2:C * 3 / 4, :], f4_down[:, C * 3 / 4:, :]),
            dim=1)
        x3 = x3_left * self.mu4 + x3_right
        x3 = self.spatial_attention3(x3)

        x4 = self.sa4(x3) # B, D, N   [8,128,4096]



        if self.training:
            x = torch.cat((x1, x2, x3, x4), dim=1)  # B, D, N   [8,512,4096]
            assist_x = torch.cat((f1, f2, f3, f4), dim=1)  # B, D, N   [8,512,4096]

            x = self.conv_fuse(x)  # B, D, N   [8,1024,4096]
            assist_x = self.conv_fuse(assist_x)  # B, D, N   [8,1024,4096]

            x_max = x.max(dim=2, keepdim=False)[0]  # B, D      [8,1024]
            assist_x_max = assist_x.max(dim=2, keepdim=False)[0]

            x_avg = x.mean(dim=2, keepdim=False)  # B, D      [8,1024]
            assist_x_avg = assist_x.mean(dim=2, keepdim=False)

            x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]
            assist_x_max_feature = assist_x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]

            x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]
            assist_x_avg_feature = assist_x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)  # B, D, N   [8,1024,4096]

            cls_label_one_hot = cls_label.view(batch_size, 1, 1)  # [8,1,1]
            cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)  # B, D, N   [8,64,4096]

            x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature),
                                         1)  # 1024 + 64   # B, D, N   [8,2112,4096]
            assist_x_global_feature = torch.cat((assist_x_max_feature, assist_x_avg_feature, cls_label_feature),
                                         1)  # 1024 + 64   # B, D, N   [8,2112,4096]

            x = torch.cat((x, x_global_feature), 1)  # 1024 * 3 + 64   # B, D, N   [8,3136,4096]
            assist_x = torch.cat((assist_x, assist_x_global_feature), 1)  # 1024 * 3 + 64   # B, D, N   [8,3136,4096]

            x = self.relu(self.bns1(self.convs1(x)))  # B, D, N   [8,512,4096]
            assist_x = self.relu(self.bns1(self.convs1(assist_x)))

            x = self.dp1(x)  # B, D, N   [8,512,4096]
            assist_x = self.dp1(assist_x)  # B, D, N   [8,512,4096]

            x = self.relu(self.bns2(self.convs2(x)))  # B, D, N   [8,256,4096]
            assist_x = self.relu(self.bns2(self.convs2(assist_x)))  # B, D, N   [8,256,4096]

            x = self.convs3(x)  # B, D, N   [8,2,4096]
            assist_x = self.convs3(assist_x)

            x = x.permute(0,2,1)
            assist_x = assist_x.permute(0, 2, 1)

            return x, assist_x


        else:

            x = torch.cat((x1, x2, x3, x4), dim=1)    # B, D, N   [8,512,4096]
            x = self.conv_fuse(x)                     # B, D, N   [8,1024,4096]
            x_max = x.max(dim=2, keepdim=False)[0]    # B, D      [8,1024]

            x_avg = x.mean(dim=2, keepdim=False)      # B, D      [8,1024]
            x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)    # B, D, N   [8,1024,4096]
            x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)    # B, D, N   [8,1024,4096]
            cls_label_one_hot = cls_label.view(batch_size,1,1)    # [8,1,1]
            cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)   # B, D, N   [8,64,4096]
            x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64   # B, D, N   [8,2112,4096]
            x = torch.cat((x, x_global_feature), 1) # 1024 * 3 + 64   # B, D, N   [8,3136,4096]
            x = self.relu(self.bns1(self.convs1(x)))                  # B, D, N   [8,512,4096]
            x = self.dp1(x)                                           # B, D, N   [8,512,4096]
            x = self.relu(self.bns2(self.convs2(x)))                  # B, D, N   [8,256,4096]
            x = self.convs3(x)                                        # B, D, N   [8,2,4096]

            x = x.permute(0, 2, 1)

            return x, None




class Leader_Point_Transformer_cls(nn.Module):
    def __init__(self, num_class=3, hidden_dim=64, guidance_dim=0, nheads=4):
        super(Leader_Point_Transformer_cls, self).__init__()
        self.num_class = num_class

        self.spatial_attention1 = SpatialTransformerLayer(hidden_dim * 2,
                                                         guidance_dim,
                                                         nheads=nheads)
        self.spatial_attention2 = SpatialTransformerLayer(hidden_dim * 2,
                                                         guidance_dim,
                                                         nheads=nheads)
        self.spatial_attention3 = SpatialTransformerLayer(hidden_dim * 2,
                                                         guidance_dim,
                                                         nheads=nheads)
        self.spatial_attention4 = SpatialTransformerLayer(hidden_dim * 2,
                                                         guidance_dim,
                                                         nheads=nheads)



        self.conv1 = nn.Conv1d(6, 128, kernel_size=1, bias=False)  # normal时=6
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)


        self.convmsf1 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf2 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf3 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.convmsf4 = nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False)
        self.bnmsf1 = nn.BatchNorm1d(128)
        self.bnmsf2 = nn.BatchNorm1d(128)
        self.bnmsf3 = nn.BatchNorm1d(128)
        self.bnmsf4 = nn.BatchNorm1d(128)
        self.mu1 = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
        self.mu2 = nn.Parameter(torch.tensor([0.15], dtype=torch.float32))
        self.mu3 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.mu4 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)



        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.2))



        self.fc1 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)



    def forward(self, x, f1, f2, f3, f4):    # x = [8,3,4096], cls_label = [8,1]
        batch_size, C, N = x.size()
        # print(x.shape)                # x = [8,6,4096]
        # print(cls_label.shape)        # cls_label = [8,1]
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N   [8,128,4096]
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N   [8,128,4096]

        x_right = torch.cat((x,f1),dim=1)
        x_right = F.relu(self.bnmsf1(self.convmsf1(x_right)))
        x_up = torch.roll(x,-1,2)
        x_dowm = torch.roll(x,1,2)
        f1_up = torch.roll(f1,-1,2)
        f1_down = torch.roll(f1,1,2)
        x_left = torch.cat((x_up[:,:C/4,:], x_dowm[:,C/4:C/2,:] , f1_up[:,C/2:C*3/4,:], f1_down[:,C*3/4:,:]),dim=1)
        x = x_left*self.mu1 + x_right
        x = self.spatial_attention1(x)

        x1 = self.sa1(x)  # B, D, N   [8,128,4096]


        x1_right = torch.cat((x1,f2),dim=1)
        x1_right = F.relu(self.bnmsf2(self.convmsf2(x1_right)))
        x1_up = torch.roll(x1,-1,2)
        x1_dowm = torch.roll(x1,1,2)
        f2_up = torch.roll(f2,-1,2)
        f2_down = torch.roll(f2,1,2)
        x1_left = torch.cat((x1_up[:,:C/4,:], x1_dowm[:,C/4:C/2,:] , f2_up[:,C/2:C*3/4,:], f2_down[:,C*3/4:,:]),dim=1)
        x1 = x1_left*self.mu2 + x1_right
        x1 = self.spatial_attention2(x1)


        x2 = self.sa2(x1) # B, D, N   [8,128,4096]


        x2_right = torch.cat((x2, f3), dim=1)
        x2_right = F.relu(self.bnmsf3(self.convmsf3(x2_right)))
        x2_up = torch.roll(x2, -1, 2)
        x2_dowm = torch.roll(x2, 1, 2)
        f3_up = torch.roll(f3, -1, 2)
        f3_down = torch.roll(f3, 1, 2)
        x2_left = torch.cat(
            (x2_up[:, :C / 4, :], x2_dowm[:, C / 4:C / 2, :], f3_up[:, C / 2:C * 3 / 4, :], f3_down[:, C * 3 / 4:, :]),
            dim=1)
        x2 = x2_left * self.mu3 + x2_right
        x2 = self.spatial_attention3(x2)


        x3 = self.sa3(x2) # B, D, N   [8,128,4096]

        x3_right = torch.cat((x3, f4), dim=1)
        x3_right = F.relu(self.bnmsf4(self.convmsf4(x3_right)))
        x3_up = torch.roll(x3, -1, 2)
        x3_dowm = torch.roll(x3, 1, 2)
        f4_up = torch.roll(f4, -1, 2)
        f4_down = torch.roll(f4, 1, 2)
        x3_left = torch.cat(
            (x3_up[:, :C / 4, :], x3_dowm[:, C / 4:C / 2, :], f4_up[:, C / 2:C * 3 / 4, :], f4_down[:, C * 3 / 4:, :]),
            dim=1)
        x3 = x3_left * self.mu4 + x3_right
        x3 = self.spatial_attention3(x3)



        x4 = self.sa4(x3) # B, D, N   [8,128,4096]



        if self.training:
            x = torch.cat((x1, x2, x3, x4), dim=1)  # B, D, N   [8,512,4096]
            assist_x = torch.cat((f1, f2, f3, f4), dim=1)  # B, D, N   [8,512,4096]

            x = self.conv_fuse(x)  # B, D, N   [8,1024,4096]
            assist_x = self.conv_fuse(assist_x)  # B, D, N   [8,1024,4096]

            x = x.max(dim=2, keepdim=False)[0]  # B, D      [8,1024]
            assist_x = assist_x.max(dim=2, keepdim=False)[0]

            x = x.view(batch_size, -1)
            assist_x = assist_x.view(batch_size, -1)

            x = self.drop1(F.relu(self.bn3(self.fc1(x))))
            assist_x = self.drop1(F.relu(self.bn3(self.fc1(assist_x))))

            x = self.drop2(F.relu(self.bn4(self.fc2(x))))
            assist_x = self.drop2(F.relu(self.bn4(self.fc2(assist_x))))


            x = self.fc3(x)
            assist_x = self.fc3(assist_x)

            x = F.log_softmax(x, -1)
            assist_x = F.log_softmax(assist_x, -1)

            return x, assist_x


        else:

            x = torch.cat((x1, x2, x3, x4), dim=1)    # B, D, N   [8,512,4096]
            x = self.conv_fuse(x)                     # B, D, N   [8,1024,4096]
            x = x.max(dim=2, keepdim=False)[0]    # B, D      [8,1024]
            x = x.view(batch_size, -1)
            x = self.drop1(F.relu(self.bn3(self.fc1(x))))
            x = self.drop2(F.relu(self.bn4(self.fc2(x))))
            x = self.fc3(x)
            x = F.log_softmax(x, -1)
            return x, None





class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c   [8,4096,32]
        x_k = self.k_conv(x)             # b, c, n    [8,32,4096]
        x_v = self.v_conv(x)             # [8,128,4096]
        energy = torch.bmm(x_q, x_k)     # b, n, n   [8,4096,4096]
        attention = self.softmax(energy) # b, n, n   [8,4096,4096]
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))    # [8,4096,4096]
        x_r = torch.bmm(x_v, attention)  # b, c, n   [8,128,4096]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) # b, c, n   [8,128,4096]
        x = x + x_r    # b, c, n   [8,128,4096]
        return x
