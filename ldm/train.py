import os
import argparse
from matplotlib import pyplot as plt
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from utils.metric import metric
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

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # ! solve warning



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
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    return init_func


def get_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.hydra_path, f"{config.job_name}.log"))
    rich_handler = RichHandler()

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False
    log.info("Successfully create rich logger")

    return log




def train(config, model, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # * init averageMeter
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    # * init lists to store loss and dice values for each epoch
    loss_avg_list = []
    dice_avg_list = []
    all_epoch_dices = []  # 添加一个列表来存储每个 epoch 的平均 Dice 值
    
    # init rich progress
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
    )

    # * set optimizer 梯度优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)

    # * set loss function 优化器让loss最小化
    from utils.loss_function import Binary_Loss, DiceLoss, cross_entropy_3D, make_one_hot

    criterion = Binary_Loss()
    # * dice_loss = DiceLoss()

    # * set scheduler strategy
    if config.use_scheduler:
        scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    # * load model
    if config.load_mode == 1:  # * load weights from checkpoint
        logger.info(f"load model from: {os.path.join(config.ckpt, config.latest_checkpoint_file)}")
        ckpt = torch.load(
            os.path.join(config.ckpt, config.latest_checkpoint_file), map_location=lambda storage, loc: storage
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.train()

    # * tensorboard writer
    writer = SummaryWriter(config.hydra_path)

    # * load datasetBs
    from dataloader import Dataset

    train_dataset = Dataset(config)
    
    #! in distributed training, the 'shuffle' must be false!
    train_loader = torch.utils.data.DataLoader(      #dataloader把数据转换为tensor
        dataset=train_dataset.queue_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    
    epochs = config.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    epoch_tqdm = progress.add_task(description="[red]epoch progress", total=epochs)
    batch_tqdm = progress.add_task(description="[blue]batch progress", total=len(train_loader))

    accelerator = Accelerator()
    # * accelerate prepare
    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)

    progress.start()
    for epoch in range(1, epochs + 1):
        progress.update(epoch_tqdm, completed=epoch)
        epoch += elapsed_epochs

        num_iters = 0

        load_meter = AverageMeter()
        train_time = AverageMeter()
        load_start = time.time()  # * initialize

        all_dices = []  # 添加一个列表来存储每个 batch 的 Dice 值
        
        for i, batch in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                progress.update(batch_tqdm, completed=i + 1)
                train_start = time.time()
                load_time = time.time() - load_start
                optimizer.zero_grad()

                x = batch["source"]["data"]
                gt = batch["gt"]["data"]
                gt_back = torch.zeros_like(gt)
                gt_back[gt == 0] = 1
                gt = torch.cat([gt_back, gt], dim=1)
                x = x.type(torch.FloatTensor).to(accelerator.device)
                gt = gt.type(torch.FloatTensor).to(accelerator.device)

                pred = model(x)
                mask = pred.argmax(dim=1, keepdim=True)
                loss = criterion(pred, gt)
                accelerator.backward(loss)
                optimizer.step()

                num_iters += 1
                iteration += 1

                _, dice = metric(gt.cpu().argmax(dim=1, keepdim=True), mask.cpu())
                all_dices.append(dice)  # 存储每个 batch 的 Dice 值

                writer.add_scalar("Training/Loss", loss.item(), iteration)
                writer.add_scalar("Training/dice", dice, iteration)

                loss_meter.update(loss.item(), x.size(0))
                dice_meter.update(dice, x.size(0))
                train_time.update(time.time() - train_start)
                load_meter.update(load_time)

                logger.info(
                    f"\nEpoch: {epoch} Batch: {i}, data load time: {load_meter.val:.3f}s , train time: {train_time.val:.3f}s\n"
                    f"Loss: {loss_meter.val}\n"
                    f"Dice: {dice_meter.val}\n"
                )

        # * clear cache
        import gc
        gc.collect()
        torch.cuda.empty_cache() 
        
        if config.use_scheduler:
            scheduler.step()
            logger.info(f"Learning rate:  {scheduler.get_last_lr()[0]}")

        epoch_dice_avg = sum(all_dices) / len(all_dices) if all_dices else 0
        all_epoch_dices.append(epoch_dice_avg)  # 存储每个 epoch 的平均 Dice 值

        logger.info(
            f"\nEpoch {epoch} used time:  {load_meter.sum+train_time.sum:.3f} s\n"
            f"Loss Avg:  {loss_meter.avg}\n"
            f"Dice Avg:  {dice_meter.avg}\n"
        )

        scheduler_dict = scheduler.state_dict() if config.use_scheduler else None
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler_dict,
                "epoch": epoch,
            },
            os.path.join(config.hydra_path, config.latest_checkpoint_file),
        )

        if epoch % config.epochs_per_checkpoint == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler_dict,
                    "epoch": epoch,
                },
                os.path.join(config.hydra_path, f"checkpoint_{epoch:04d}.pt"),
            )
        # save every epoch loss_avg and dice_avg 
        loss_avg_list.append(loss_meter.avg)
        dice_avg_list.append(dice_meter.avg)

    # train finished, draw loss and dice curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(loss_avg_list, label='Train Loss', color='blue', marker='o')
    plt.plot(dice_avg_list, label='Train Dice', color='orange', marker='x')
    plt.title('Training Loss and Dice Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend() # 显示图例
    plt.grid() # 显示网格
    plt.savefig(os.path.join(config.hydra_path, 'train_loss_dice_curve.png'))  # 保存图像
    plt.show()  # 展示图像

    writer.close()

    # 计算并打印所有 epoch 的平均 Dice 值
    if all_epoch_dices:
        overall_dice_avg = sum(all_epoch_dices) / len(all_epoch_dices)
        logger.info(f"Overall Average Dice: {overall_dice_avg:.4f}")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    #配置config
    if isinstance(config.patch_size, str):
        assert (
            len(config.patch_size.split(",")) <= 3
        ), f'patch size can only be one str or three str but got {len(config.patch_size.split(","))}'
        if len(config.patch_size.split(",")) == 3:
            config.patch_size = tuple(map(int, config.patch_size.split(",")))
        else:
            config.patch_size = int(config.patch_size)

    # * model selection
    if config.network == "res_unet":  #2017
        from models.three_d.residual_unet3d import UNet

        model = UNet(in_channels=config.in_classes, n_classes=config.out_classes, base_n_filter=32)
    elif config.network == "unet":    #2016
        from models.three_d.unet3d import UNet3D  # * 3d unet

        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    elif config.network == "er_net":
        from models.three_d.ER_net import ER_Net

        model = ER_Net(classes=config.out_classes, channels=config.in_classes)
    elif config.network == "re_net":
        from models.three_d.RE_net import RE_Net

        model = RE_Net(classes=config.out_classes, channels=config.in_classes)

    elif config.network == "mlla_unet3d":
        from models.three_d.mlla_unet3d import MLLA_UNet  # * 3d unet

        model = MLLA_UNet(in_chans=config.in_classes, num_classes=config.num_classes, embed_dim=48)

    elif config.network == "densenet3d": 
        from models.three_d.densenet3d import SkipDenseNet3D  # * 3d unet

        model = SkipDenseNet3D(in_channels=config.in_classes, classes=config.out_classes)

    elif config.network == "csrnet": 
        from models.three_d.csrnet import CSRNet  # * 3d unet

        model = CSRNet(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)

    elif config.network == "vt_unet": 
        from models.three_d.vt_unet import SwinTransformerSys3D  # * 3d unet

        model = SwinTransformerSys3D(in_chans=config.in_classes, num_classes=config.out_classes, embed_dim=48)
    
    elif config.network == "vtnet": 
        from models.three_d.vtnet import VTUNet  # * 3d unet

        model = VTUNet(input_dim=config.in_classes, num_classes=config.out_classes, embed_dim=48)
    
    elif config.network == "DenseVoxelNet": 
        from models.three_d.densevoxelnet3d import DenseVoxelNet  # * 3d unet

        model = DenseVoxelNet(in_channels=config.in_classes, classes=config.out_classes)

    elif config.network == "IS":    #2016
        from models.three_d.IS import UNet3D  # * 3d unet

        model = UNet3D(in_channels=config.in_classes, out_channels=config.out_classes, init_features=32)
    
    elif config.network == "unetr":    #2021
        from models.three_d.unetr import UNETR  # * 3d unet

        model = UNETR(input_dim=config.in_classes, output_dim=config.out_classes)
        #model = UNETR()

    elif config.network == "FusionNet":    #2016
        from models.three_d.FusionNet import FusionNet  # * 3d unet

        model = FusionNet(in_channels=config.in_classes, out_channels=config.out_classes, unet_init_features=32)
    # * init model weights
    model.apply(weights_init_normal(config.init_type))
    
    # * create logger
    logger = get_logger(config)
    info = "\nParameter Settings:\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    logger.info(info)
    #* train
    train(config, model, logger)

    
    logger.info(f"tensorboard file saved in:{config.hydra_path}")


if __name__ == "__main__":
    main()
