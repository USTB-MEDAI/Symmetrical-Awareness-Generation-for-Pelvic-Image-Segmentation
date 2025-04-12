import hydra
import numpy as np
import torch
from pathlib import Path 
from torch.utils.data import DataLoader, Dataset

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    LoadImage,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    ScaleIntensityd,
    NormalizeIntensityd,
    Spacingd,
    EnsureType,
    Resized,
    SaveImage,
)
import monai.transforms as mt
import pandas as pd
import random
from monai.networks.layers import Norm
from monai.data import CacheDataset, list_data_collate, decollate_batch, Dataset


def prepare_dataset(data_path, resize_size, img_resize_size=None, cond_path=None, split="train",cond_nums=[1]):
    """
    Prepare dataset for training
    data_path: str, path to nii data(3D)
    cond_path: str, path to x-ray 2d png images, if None means only conduct autoencoder process
    resize_size: tuple, (x, y, z)
    split: str, "train" or "val"
    """
    #确保提取出来的 data_list 和 cond_list 按照原来的顺序一一对应。我们可以通过只加载一次CSV文件来实现这一点
    img_column_name = 'image'  # 要提取的列名
    # 加载CSV文件
    df = pd.read_csv(data_path)
    # 提取指定列的数据并转换为列表
    data_list = [Path(p) for p in df[img_column_name]]
    print(f"data_list_len:{len(data_list)}")
    
    
    #读取Mask数据
    mask_path = "/home/syz/Versedata/Lumbar/label_cropped/cropped_spine_mask_49.nii.gz"
    mask_list = Path(mask_path)
    
    if cond_path:
        cond_column_name = 'Text'  # 要提取的列名
        # 加载CSV文件
        # df = pd.read_csv(data_path)
        # 提取指定列的数据并转换为列表
        cond_list = df[cond_column_name]
        print(f"cond_path_len:{len(cond_list)}")

    # * create data_dicts, a list of dictionary with keys "image" and "cond",  cond means x-ray 2d png image
    data_dicts = []
    if cond_path:
        for i, (image, cond) in enumerate(zip(data_list, cond_list)):
            tmp = {"image": image}
            tmp["mask"] = mask_list
            # print(f"tmp:{tmp}")
            # print(f"image:{image}")
            # print(f"cond:{cond}")
            # # cond_png = list(sorted(Path(cond).glob("*.nii.gz*")))
            # print(f"cond_type:{type(cond)}")
            # cond_png = list(cond)
            # print(f"cond_png_len:{len(cond_png)}")
            if 1 in cond_nums:               # 一个image（source）对应一个cond（label）
                tmp["cond1"] = cond
            # if 2 in cond_nums:
            #     tmp["cond2"] = cond_png[1]
            # if 3 in cond_nums:
            #     tmp["cond3"] = cond_png[2]
            data_dicts.append(tmp)
            #print(f"data_dicts:{data_dicts}")
    else:
        for image in data_list:
            tmp = {"image": image}
            data_dicts.append(tmp)
    #打乱数据
    random.seed(16)
    random.shuffle(data_dicts)
    
    #cond_keys = []
    load_keys = ["image"]
    load_keys.append("mask")
    # if 1 in cond_nums:
    #     cond_keys.append("cond1")
    #     load_keys.append("cond1")
    # if 2 in cond_nums:
    #     cond_keys.append("cond2")
    #     load_keys.append("cond2")
    # if 3 in cond_nums:
    #     cond_keys.append("cond3")
    #     load_keys.append("cond3")
    # print(cond_keys)
    # load_keys.append("image")
    # print(load_keys)
     
    if split == "train":
        data_dicts = data_dicts[: int(len(data_dicts) * 0.8)]
    else:
        data_dicts = data_dicts[int(len(data_dicts) * 0.8) :]
    #print(f"data_dicts: {[data_dict['image'] for data_dict in data_dicts]}")
    print(f"data_dicts:{data_dicts}")
    print(f"data_dicts_len:{len(data_dicts)}")
    # if split == "train":
    #     data_dicts = data_dicts[: int(len(data_dicts) * 0.7)]
    # elif split == "test":
    #     data_dicts = data_dicts[int(len(data_dicts) * 0.8) : ]
    # elif split == "val":
    #     data_dicts = data_dicts[int(len(data_dicts) * 0.7) : int(len(data_dicts) * 0.8)]

    set_determinism(seed=0)

    if cond_path:
        train_transforms = Compose(
            [
                LoadImaged(keys=load_keys, ensure_channel_first=True),
                
                # Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["image"], spatial_size=resize_size),
                Resized(keys=["mask"], spatial_size=resize_size),
                # NormalizeIntensityd(keys=used_keys),
                # NormalizeIntensityd(keys=["image"]),
                ############## else ###############
                # ScaleIntensityd(keys=["imgae"]),
                ############## Fei  ###############
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
                #ScaleIntensityd(keys=cond_keys, minv=-1, maxv=1), #如果是label数据需要删掉这句
            ]
        )
    else:
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["image"], spatial_size=resize_size),
                ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),  # 如果是label数据需要删掉这句
            ]
        )
    if split == "train":
        train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, num_workers=0)
    else:
        train_ds = Dataset(data=data_dicts, transform=train_transforms)
    shuffle = False if split == "test" else True
    train_dl = DataLoader(train_ds, batch_size=1, num_workers=4, shuffle=shuffle)
    return train_dl


@hydra.main(config_path="../conf", config_name="config/autoencoder.yaml", version_base="1.3")
def main(config):
    config = config["config"]
    # train_dl = prepare_dataset(
    #     data_path=config.data_path, resize_size=config.resize_size, cond_path=config.cond_path, split="test"
    # )
    test_dl = prepare_dataset(
        cond_path="/home/syz/Versedata/VerseT2Idata_v2.csv",
        #cond_path=None,
        data_path="/home/syz/Versedata/VerseT2Idata_v2.csv",
        resize_size=config.resize_size,
        img_resize_size=config.resize_size,
        split="train",
    )
    print(f"test_dl_len:{len(test_dl)}")
    
    
    for i in test_dl:
        # print(i["image"])
        # continue
        # print(i["cond1"].shape)
        #cond1 = i["cond1"]
        # cond1 = i["cond1"].permute(0, 2, 3, 1)
        # cond1 = cond1 * 255
        # cond = cond[:, :, :, :3]
        # print(f"max value {max(cond)}, min value {min(cond)}")
        # cond = cond * 255
        # print(cond.shape)
        img = i["image"]
        # img = img.squeeze(0)
        # img = img * 255
        mask = i["mask"]
        img_cat1 = torch.cat([img, mask], dim=1)
        print(img_cat1.shape)
        img = (img + 1) * 127.5
        img_cat2 = torch.cat([img, mask], dim=1)
        print(img_cat2.shape)
        print(img.shape)
        #img = img.reshape(1, 128, 128, 128)
        #cond1 = cond1.reshape(1, 128, 128, 128)
        saver_origin = SaveImage(
            output_dir="./",
            output_ext=".nii.gz",
            output_postfix="cache",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        saver_origin(img)
        saver_cat1 = SaveImage(
            output_dir="./",
            output_ext=".nii.gz",
            output_postfix="cat1",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        saver_cat1(img_cat1)
        saver_cat2 = SaveImage(
            output_dir="./",
            output_ext=".nii.gz",
            output_postfix="cat2",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        saver_cat2(img_cat2)
        #print(f"cond1_len: {len(cond1)}")
        break
        '''
        saver = SaveImage(
            output_dir="./",
            output_ext=".nii.gz",
            output_postfix="PIL",
            output_dtype=np.uint8,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        img = saver(cond1)
        break
        '''


def test_save_image():
    # img = LoadImage()
    path = "/disk/ssy/data/drr/feijiejie/all/LNDb-0210.nii"
    trans = Compose(
        [
            LoadImaged(keys="image", ensure_channel_first=True),
            # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            Resized(keys="image", spatial_size=(128, 128, 128)),
            # ScaleIntensityd(keys="image"),
            ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
        ]
    )
    d = {"image": path}
    img = trans(d)
    img = img["image"]
    # print(img.shape)
    # print(img.affine)
    # print(img.meta.keys())
    img = (img + 1) * 127.5
    # print(img.shape)
    saver_origin = SaveImage(
        output_dir="./",
        output_ext=".nii.gz",
        output_postfix="origin",
        separate_folder=False,
        output_dtype=np.uint8,
        # scale=255,
        resample=False,
        squeeze_end_dims=True,
        writer="NibabelWriter",
    )
    saver_origin(img)


if __name__ == "__main__":
    main()
    #test_save_image()
