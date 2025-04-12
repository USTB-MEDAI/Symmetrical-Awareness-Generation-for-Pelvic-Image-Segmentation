from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import hydra
import os
import logging
from rich.logging import RichHandler
import seaborn as sns

# 复用训练代码中的组件
from TextClassHead import Classifier, get_subjects, get_logger, weights_init_normal

import matplotlib
print(matplotlib.get_cachedir()) 

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    
    # 创建日志记录器
    logger = get_logger(config)
    logger.info("Starting testing process...")
    
    # 初始化模型
    model = Classifier(config.input_dim, config.num_classes)
    model.apply(weights_init_normal(init_type='normal'))
    
    # 加载训练好的权重
    checkpoint_path = os.path.join(config.hydra_path, config.latest_checkpoint_file)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # 设置为评估模式
    model.eval()
    
    # 加载测试数据集
    config.job_name = "predict"  # 强制使用测试集
    test_dataset = get_subjects(config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 存储预测结果
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            text_emb = batch["Text"].float()
            labels = batch["label"]
            
            text_emb = text_emb.type(torch.FloatTensor).to('cpu')
            labels = torch.tensor(labels).to('cpu')
            # 前向传播
            logits = model(text_emb).to('cpu')
            preds = torch.argmax(logits, dim=1)
            
            # 收集结果
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"\nTest Accuracy: {accuracy:.4f}")
    
    # 分类报告
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, 
                                     target_names=["sacrum", "left hip", "right hip"]))
    
    # 混淆矩阵
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\n{cm}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12))
    #修改字体为罗马体
    plt.rcParams['font.family'] = 'Times New Roman'
    #字体加粗
    plt.rcParams['font.weight'] = 'bold'
    #只增大数值结果的字体
    plt.rcParams['font.size'] = 45
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["sacrum", "left hip", "right hip"],
                yticklabels=["sacrum", "left hip", "right hip"])
    # sns.set(font_scale=1.5) # 将混淆矩阵中的数字字体变大
    plt.xlabel('Predicted label', fontsize=45, fontweight='bold')
    plt.ylabel('True label', fontsize=45, fontweight='bold')
    plt.title('Confusion matrix', fontsize=45, fontweight='bold')
    plt.savefig('confusion_matrix.png')  # 保存图片到文件
    plt.show()

if __name__ == "__main__":
    main()