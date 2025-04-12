from venv import logger
import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertModel, BertTokenizer
from CoordinateEmbeder import CoordinateEmbedder
from TextEncoder_UniCLIP import Text_Encoder
from dataset.monai_nii_dataset import prepare_dataset
import numpy as np

class CoordinatePredictor(nn.Module):
    def __init__(self, coord_dim=3, embed_dim=512):
        super().__init__()
        # 可训练的坐标编码器
        self.coord_embedder = CoordinateEmbedder(coord_dim, embed_dim)
        
            
        # 将坐标特征对齐到文本空间
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

    def forward(self, coords):
        # 坐标编码 [B,1,512] -> [B,512]
        coord_features = self.projection(self.coord_embedder(coords).squeeze(1))
        
        return coord_features
    

def contrastive_loss(logits, temperature=0.07):
    # 计算对比学习损失
    labels = torch.arange(logits.size(0)).to(logits.device)
    return nn.CrossEntropyLoss()(logits / temperature, labels)

def train():
    device = torch.device("cuda:1")
    # device = torch.device("cpu")
    # 模型配置
     # 冻结的文本编码器
    text_encoder = Text_Encoder(
                                model_name = 'ViT-B-16-quickgelu', 
                                pretrained_weights = "/disk/SYZ/UniMed-CLIP-main/unimed_clip_vit_b16.pt", 
                                device = 'cuda:1', 
                                text_encoder_name = "/disk/SYZ/UniMed-CLIP-main/BiomedNLP-BiomedBERT-base-uncased-abstract", 
                                context_length=77, 
                                freeze=True, 
                                checkpoint_path="/disk/SYZ/Xray-Diffsuion/logs/UniCLIP/TextClassHead-2025-03-28/04-07-10/latest_checkpoint.ckpt"
                            )
 
    model = CoordinatePredictor().to(device)
    optimizer = torch.optim.Adam(model.coord_embedder.parameters(), lr=1e-5)
        
    # 数据加载
    dataloader = prepare_dataset(data_path = "/disk/SYZ/Xray-Diffsuion/datacsv/TotalPelvic_600_v2.csv", 
                                 resize_size=[128, 128, 128], 
                                 img_resize_size=None, 
                                 cond_path="/disk/SYZ/Xray-Diffsuion/datacsv/TotalPelvic_600_v2.csv", 
                                 split="train",
                                 cond_nums=[1],
                                 bs=4, 
                                 fast_mode=False)

    # 训练循环
    model.train()
    for epoch in range(60):
        total_loss = 0
        for batch in dataloader:
            # 数据准备
            # coords = batch["coord"]
            text_data = batch["cond1"]
            coords = torch.tensor(batch["coord"]).clone().detach().float().to(device)
            # 文本编码 [B,512]
            text_features = text_encoder.forward(text_data).squeeze(1).to(device)
            # text_data = torch.tensor([int(x) for x in batch["cond1"]]).long().to(device)
            # 前向传播
            coord_emb = model(coords)
            
            # 计算相似度矩阵
            logits = torch.matmul(coord_emb.to(device), text_features.t().to(device))  # [B,B]
            
            # 对称对比损失
            loss = (contrastive_loss(logits) + contrastive_loss(logits.t())) / 2
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        #训练日志,指定日志保存路径文件
        logger.info(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        # 保存模型
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        # print("coords device:", coords.device)
        # print("model device:", next(model.parameters()).device)
        # 保存模型
        if (epoch+1) % 10 == 0:

            torch.save(model.state_dict(), f"/disk/SYZ/Xray-Diffsuion/logs/Coords/last_checkpoint.ckpt")
            device = torch.device("cuda:1")
            # model.load_state_dict(torch.load("/disk/SYZ/Xray-Diffsuion/logs/Coords/last_checkpoint.ckpt"))
            # 验证集加载
            val_loader = prepare_dataset(data_path = "/disk/SYZ/Xray-Diffsuion/datacsv/TotalPelvic_600_v2.csv", 
                                        resize_size=[128, 128, 128], 
                                        img_resize_size=None, 
                                        cond_path="/disk/SYZ/Xray-Diffsuion/datacsv/TotalPelvic_600_v2.csv", 
                                        split="val",
                                        cond_nums=[1],
                                        bs=8, 
                                        fast_mode=False)
            # 验证
            evaluate(model, val_loader)

def evaluate(model, val_loader):
    model.eval()
    device = torch.device("cuda:1")
    text_encoder = Text_Encoder(
                                model_name = 'ViT-B-16-quickgelu', 
                                pretrained_weights = "/disk/SYZ/UniMed-CLIP-main/unimed_clip_vit_b16.pt", 
                                device = 'cuda:1', 
                                text_encoder_name = "/disk/SYZ/UniMed-CLIP-main/BiomedNLP-BiomedBERT-base-uncased-abstract", 
                                context_length=77, 
                                freeze=True, 
                                checkpoint_path="/disk/SYZ/Xray-Diffsuion/logs/UniCLIP/TextClassHead-2025-03-28/04-07-10/latest_checkpoint.ckpt"
                            )
    correct = 0
    coord_errors = []
    with torch.no_grad():
        for batch in val_loader:
            # 数据准备
            text_data = batch["cond1"]
            coords = torch.tensor(batch["coord"]).clone().detach().float().to(device)
            # 文本编码 [B,512]
            text_features = text_encoder.forward(text_data).squeeze(1).to(device)
            coord_emb = model(coords)
            # 计算最近邻匹配
            similarities = torch.matmul(coord_emb.to(device), text_features.t().to(device))
            predictions = similarities.argmax(dim=1)
            correct += (predictions == torch.arange(len(predictions)).to(device)).sum()
            #坐标误差
            nearest_idx = similarities.argmax(dim=0)
            pred_coords = coords[nearest_idx]
            mse = torch.mean((coords - pred_coords)**2, dim=1)
            coord_errors.extend(mse.cpu().tolist())
    
    accuracy = correct / len(val_loader.dataset)
    avg_error = sum(coord_errors) / len(coord_errors)
    print(f"Matching Accuracy: {accuracy:.2%}")
    print(f"  Mean Coordinate MSE: {avg_error:.4f}")
    logger.info(f"Matching Accuracy: {accuracy:.2%}")
    logger.info(f"  Mean Coordinate MSE: {avg_error:.4f}")

if __name__ == "__main__":
    

    eval = False

    if eval == False:
        train()
        # # evaluate()
            #加载模型
    
    else:
        device = torch.device("cuda:1")
        model = CoordinatePredictor().to(device)
        model.load_state_dict(torch.load("/disk/SYZ/Xray-Diffsuion/logs/Coords/last_checkpoint.ckpt"))
        # 验证集加载
        test_loader = prepare_dataset(data_path = "/disk/SYZ/Xray-Diffsuion/datacsv/TotalPelvic_600_v2.csv", 
                                    resize_size=[128, 128, 128], 
                                    img_resize_size=None, 
                                    cond_path="/disk/SYZ/Xray-Diffsuion/datacsv/TotalPelvic_600_v2.csv", 
                                    split="test",
                                    cond_nums=[1],
                                    bs=8, 
                                    fast_mode=False)
        # 验证
        evaluate(model, test_loader)