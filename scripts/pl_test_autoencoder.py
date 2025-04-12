import hydra
from rich.logging import RichHandler
from torch.utils.data import DataLoader

# from datasets.med_3Ddataset import ImageDataset

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from dataset.monai_nii_dataset import prepare_dataset
from dataset.med_3Ddataset import ImageDataset
import lightning as pl
from ldm.autoencoderkl.autoencoder import AutoencoderKL
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset.monai_nii_dataset import prepare_dataset


@hydra.main(config_path="../conf", config_name="config")
def train(config):
    config = config["config"]

    # * dataset and dataloader
    test_dl = prepare_dataset(data_path=config.data_path, resize_size=config.resize_size, split="test")
    # val_dataset = ImageDataset(config, split="val")
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=True,
    #     num_workers=config.num_workers,
    #     batch_size=config.batch_size,
    # )

    # * model
    model = AutoencoderKL(save_path=config.hydra_path, **config["model"])
    model.init_from_ckpt("/home/syz/Xray-Diffsuion/logs/autoencoder/pl_train_autoencoder-2025-02-18/17-33-35/pl_train_autoencoder-epoch140-val_rec_loss0.00.ckpt")

    # * test model
    # config["trainer"].devices = 1
    trainer = pl.Trainer(**config["trainer"])
    trainer.test(model=model, dataloaders=test_dl)


if __name__ == "__main__":
    train()
