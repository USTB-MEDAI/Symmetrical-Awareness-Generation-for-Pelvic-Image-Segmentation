from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from TextEncoder import FrozenCLIPEmbedder
from TextEncoder_UniCLIP import Text_Encoder

import os
import argparse
from matplotlib import pyplot as plt
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

# from logger import create_logger
from timm.utils import AverageMeter
from accelerate import Accelerator

# from utils import yaml_read
# from utils.conf_base import Default_Conf
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
import logging
from rich.logging import RichHandler
import hydra


def weights_init_normal(init_type):
    def init_func(m):
        classname = m.__class__.__name__
        gain = 0.02

        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    return init_func


def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(
        config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log


def get_subjects(config):
    """
    @description: get the subjects for normal training
    """
    # 确保提取出来的 data_list 和 cond_list 按照原来的顺序一一对应。我们可以通过只加载一次CSV文件来实现这一点
    img_column_name = 'text'  # 要提取的列名
    # 加载CSV文件
    df = pd.read_csv(config.pred_data_path)
    # 提取指定列的数据并转换为列表
    data_list = [str(Path(p)) for p in df[img_column_name]]

    print(f"data_list_len:{len(data_list)}")

    # 读取Mask数据
    mask_column_name = "label"
    # mask_list = [str(Path(p)) for p in df[mask_column_name]]
    mask_list = df[mask_column_name]
    print(f"label_list_len:{len(mask_list)}")

    subjects = []    

    # 划分数据集
    # 使用分层划分来保持类别分布的均衡性
    # mask_list 中的值是类别标签
    train_img, test_img, train_mask, test_mask = train_test_split(
        data_list, mask_list, test_size=0.2, stratify=mask_list, random_state=42
    )

    # 根据 job_name 决定使用哪部分数据
    if "TextClassHead" in config.job_name:
        img_path = train_img
        gt_path = train_mask
    else:
        img_path = test_img
        gt_path = test_mask
    print(f"img_path_len:{len(img_path)}")
    # x_generator  = sorted(img_path.glob("*.nii.gz"))#glob函数用于查找符合特定模式的文件，返回文件/文件夹名
    # gt_generator = sorted(gt_path.glob("*.nii.gz"))#sorted将列表中的元素进行排序
    # print('x_generator:', x_generator)
    #####################################
    
    text_encoder = FrozenCLIPEmbedder(version=config.version,
                                      device=torch.device(config.device),
                                      max_length=config.max_length)
    # text_encoder = Text_Encoder(model_name = config.model_name, 
    #                            pretrained_weights = config.pretrained_weights, 
    #                            device = torch.device(config.device),
    #                            text_encoder_name = config.text_encoder_name, 
    #                            context_length = config.context_length,
    #                            freeze=config.freeze)

    for i, (source, gt) in enumerate(zip(img_path, gt_path)):  # zip将两个列表中的元素打包成元组（source,gt）配对
        # print(f"source:{source}, gt:{gt}")
        source_image = text_encoder.forward(source)  # 假设source_path是.nii文件
        Mapping = {"cervical vertebrae": 0, "thoracic vertebrae": 1,
                   "lumbar vertebrae": 2, "sacrum": 3, "left hip": 4, "right hip": 5}
        if config.num_classes == 3: #and (Mapping[gt] == 3 or Mapping[gt] == 4 or Mapping[gt] == 5):
            Mapping = {"sacrum": 0, "left hip": 1, "right hip": 2}
            tmp = {"Text": source_image}
            if gt in Mapping:
                tmp["label"] = Mapping[gt]
            else:
                tmp["label"] = gt
            # print(f"tmp:{tmp}")
            subjects.append(tmp)
        elif config.num_classes == 6:
            tmp = {"Text": source_image}
            tmp["label"] = Mapping[gt]
            subjects.append(tmp)
        else:
            continue
    print(f'subjects_len:{len(subjects)}')
    # 打乱数据
    random.seed(23)
    random.shuffle(subjects)
    return subjects


# 示例数据（替换为实际数据）
device = "cpu"

# 添加分类头
class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x 

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train(config, model, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # * init averageMeter
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # * init lists to store loss and dice values for each epoch
    loss_avg_list = []
    acc_avg_list = []
    all_epoch_dices = []  # 添加一个列表来存储每个 epoch 的平均 Dice 值

    loss_1epoch_mean_list = []
    acc_1epoch_mean_list = []
    
    # init rich progress
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )
    # * 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    # * 损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练
    model.train()

    # * 加载数据集
    dataset = get_subjects(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # accelerator = Accelerator()
    # * accelerate prepare
    # train_loader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            text_emb = batch["Text"]
            labels = batch["label"]

            # Tensor 在CPU上进行运算
            text_emb = text_emb.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            # one-hot编码
            labels = torch.nn.functional.one_hot(
                labels, num_classes=config.num_classes).float()

            # 分类预测
            logits = model(text_emb).to(device)
            # print(f"logits:{logits.shape}, labels:{labels.shape}")
            # print(f"logits:{logits.shape}, labels:{labels.shape}")
            loss = criterion(logits, labels)
            acc = (logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean()

            # 记录 loss 和 acc
            loss_meter.update(loss.item(), n=text_emb.size(0))
            acc_meter.update(acc, n=text_emb.size(0))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg_list.append(loss_meter.avg)
            acc_avg_list.append(acc_meter.avg)

            # 打印日志
            if (batch_idx + 1) % config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx+1}, Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.4f}")
        
        loss_1epoch_mean = sum(loss_avg_list)/len(loss_avg_list)
        acc_1epoch_mean = sum(acc_avg_list)/len(acc_avg_list)
        
        loss_1epoch_mean_list.append(loss_1epoch_mean)
        acc_1epoch_mean_list.append(acc_1epoch_mean)
        
        # 保存模型
        if (epoch + 1) % config.save_interval == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(config.hydra_path, config.latest_checkpoint_file),
            )
    # 记录平均acc值
    acc_epoch_mean = sum(acc_avg_list) / len(acc_avg_list)
    loss_epoch_mean = sum(loss_avg_list) / len(loss_avg_list)
    all_epoch_dices.append(acc_epoch_mean)
    logger.info(f"Epoch {epoch}\n"
                f"Loss: {loss_epoch_mean:.4f}\n"
                f"Acc: {acc_epoch_mean:.4f}\n")
    
    plot_loss_acc(loss_1epoch_mean_list, acc_1epoch_mean_list, config.num_epochs)


def plot_loss_acc(all_loss_avg, all_acc_avg, all_epoch_dices):
    epochs = range(1, all_epoch_dices + 1)

    plt.figure(figsize=(12, 4))

    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, all_loss_avg, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, all_acc_avg, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig("loss_acc.png")
    plt.show()

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    model = Classifier(config.input_dim, config.num_classes)
    model.apply(weights_init_normal(init_type='normal'))

    # create logger
    logger = get_logger(config)
    logger.info(f"Config: {config}")
    # * train
    train(config, model, logger)


if __name__ == "__main__":
    main()
