import torch
import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# class CoordinateEmbedder(nn.Module):
#     def __init__(self, coord_dim=3, embed_dim=512):
#         super().__init__()
#         # 每个坐标轴独立编码
#         self.x_embed = nn.Linear(1, embed_dim//2)
#         self.y_embed = nn.Linear(1, embed_dim//4)
#         self.z_embed = nn.Linear(1, embed_dim//4)
        
#     def forward(self, coord):
#         # coord: [B, 3]
#         # print(coord.shape)
#         # print(f"coord: {coord}")
#         # print(f"type(coord): {type(coord)}")
#         x_enc = torch.sin(self.x_embed(coord[:,0:1]))  # 周期编码增强空间敏感性
#         y_enc = torch.cos(self.y_embed(coord[:,1:2]))
#         z_enc = torch.sin(self.z_embed(coord[:,2:3]))
#         return torch.cat([x_enc, y_enc, z_enc], dim=1).unsqueeze(1)  # [B, 1 , embed_dim]

class CoordinateEmbedder(nn.Module):
    def __init__(self, coord_dim=3, embed_dim=512):
        super().__init__()
        # 增强x轴编码能力
        self.x_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )
        # yz轴联合编码
        self.yz_encoder = nn.Sequential(
            nn.Linear(2, 256),
            nn.SiLU(),
            nn.Linear(256, 256))
        
    def forward(self, coord):
        # print(f"coord: {coord.device}")  #cuda:1
        x_feat = self.x_encoder(coord[:,0:1]).to(coord.device)  # [B,256]
        yz_feat = self.yz_encoder(coord[:,1:3]).to(coord.device) # [B,256]
        
        # # 显式方向标记
        # left_marker = (coord[:,0] < -0.01).float().view(-1,1)   #之前表述为<-0.5，导致误导，但实际上应该是<-0.01  #改了之后更差了
        # right_marker = (coord[:,0] > 0.4).float().view(-1,1) #之前表述为>0.5，导致误导，但实际上应该是>0.4  #改了之后更差了
        
        return torch.cat([
            x_feat , #+ left_marker - right_marker,  # 左: +1, 右: -1, 中间: 0
            yz_feat
        ], dim=-1).unsqueeze(1)  # [B,1,512]

