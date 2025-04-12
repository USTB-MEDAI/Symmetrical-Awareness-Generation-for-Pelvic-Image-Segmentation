import hydra
import numpy as np
import torch
from pathlib import Path 
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

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
    MapTransform,
)
import monai.transforms as mt
import pandas as pd
import random
from monai.networks.layers import Norm
from monai.data import CacheDataset, list_data_collate, decollate_batch, Dataset, PersistentDataset

#替换DataLoader为MultiEpochsDataLoader，使得训练数据加快
class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)



class ConcatenateImageAndMask(MapTransform):
    def __init__(self, keys, concat_key='image', dim=0):
        super().__init__(keys)
        self.concat_key = concat_key
        self.dim = dim

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]
        mask = d[self.keys[1]]
        concatenated_img = torch.cat([img, mask], dim=self.dim)
        d[self.concat_key] = concatenated_img
        return d

def prepare_dataset(data_path, resize_size, img_resize_size=None, cond_path=None, split="train",cond_nums=[1],bs=1,fast_mode=False):
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
    mask_column_name = "mask"
    mask_list = [Path(p) for p in df[mask_column_name]]
    print(f"mask_list_len:{len(mask_list)}")

    if cond_path:
        cond_column_name = 'text'  # 要提取的列名
        # 加载CSV文件
        # df = pd.read_csv(data_path)
        # 提取指定列的数据并转换为列表
        cond_list = df[cond_column_name]
        print(f"cond_path_len:{len(cond_list)}")

        #读取coord数据
        coord_column_name = "coord"
        coord_list = df[coord_column_name]
        #把每一个元素都转换为list
        coord_list = [eval(coord) for coord in coord_list]
        #把每一个元素中的每一个元素依次提取出x,y,z，然后cat到一起
        coord_list = [torch.cat([torch.tensor(coord).unsqueeze(0) for coord in coord_list_i]) for coord_list_i in coord_list]
        print(f"coord_list_len:{len(coord_list)}")
        # print(f"cood_list_type:{type(coord_list)}")
        # print(f"cood_list_0:{coord_list[0]}")
        # print(f"cood_list_e:{type(coord_list[0])}")
        
    label_column_name = 'label'  # 要提取的列名
    label_list = df[label_column_name]
    print(f"label_list_len:{len(label_list)}")

    # * create data_dicts, a list of dictionary with keys "image" and "cond",  cond means x-ray 2d png image
    data_dicts = []
    dala_label = []
    if cond_path:
        for i, (image, cond, mask, label, coord) in enumerate(zip(data_list, cond_list, 
                                                    mask_list, label_list, coord_list)):
            tmp = {"raw_image": image}
            tmp["mask"] = mask
            # print(f"tmp:{tmp}")
            # print(f"image:{image}")
            # print(f"cond:{cond}")
            # # cond_png = list(sorted(Path(cond).glob("*.nii.gz*")))
            # print(f"cond_type:{type(cond)}")
            # cond_png = list(cond)
            # print(f"cond_png_len:{len(cond_png)}")
            if 1 in cond_nums: # 一个image（source）对应一个cond（label）
                if not isinstance(cond, str):
                    cond = str(cond)
                tmp["cond1"] = cond
                tmp['label'] = label
                tmp['coord'] = coord
            # if 2 in cond_nums:
            #     tmp["cond2"] = cond_png[1]
            # if 3 in cond_nums:
            #     tmp["cond3"] = cond_png[2]
            data_dicts.append(tmp)
            dala_label.append(label)
            if fast_mode and i > 16:
                break
            #print(f"data_dicts:{data_dicts}")
    else:
        for i, (image, mask, label) in enumerate(zip(data_list, mask_list, label_list)):
            tmp = {"raw_image": image}
            tmp["mask"] = mask
            tmp['label'] = label
            data_dicts.append(tmp)
            dala_label.append(label)
            if fast_mode:
                break
    # 打乱数据
    random.seed(16)
    random.shuffle(data_dicts)
    
    # cond_keys = []
    load_keys = ["raw_image"]
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
    
    data_train, data_temp, _, label_temp = train_test_split(data_dicts, dala_label, test_size=0.3, stratify=dala_label, random_state=123)

    data_val, data_test, _, _ = train_test_split(data_temp, label_temp, test_size=2/3, stratify=label_temp, random_state=123)
    
    #Train/Val/Test = 70%/10%/20%
    if split == "train" and not fast_mode:
        data_dicts = data_train
    elif split == "test" and not fast_mode:
        # data_dicts = data_dicts[int(len(data_dicts) * 0.7): int(len(data_dicts) * 0.9)]
        data_dicts = data_test
    elif split == "val" and not fast_mode:
        data_dicts = data_val
    else:
        data_dicts = data_dicts

    #输出data_dicts保存到一个csv文件中
    df = pd.DataFrame(data_dicts)
    df.to_csv(f"data_dicts_{split}.csv", index=False)
    
    #计算data_dicts中每个类别的数量
    class_count = {}
    for data_dict in data_dicts:
        label = data_dict["label"]
        if label not in class_count:
            class_count[label] = 1
        else:
            class_count[label] += 1
    print(f"class_count:{class_count}")

    # print(f"data_dicts: {[data_dict['image'] for data_dict in data_dicts]}")
    # print(f"data_dicts:{data_dicts}")
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
                LoadImaged(keys=load_keys, ensure_channel_first=True), #确保通道维度在最前面
                
                # Orientationd(keys=["image"], axcodes="RAS"),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["raw_image"], spatial_size=resize_size),
                Resized(keys=["mask"], spatial_size=resize_size),
                # NormalizeIntensityd(keys=used_keys),
                # NormalizeIntensityd(keys=["image"]),
                ############## else ###############
                # ScaleIntensityd(keys=["imgae"]),
                ############## Fei  ###############

                ScaleIntensityRanged(keys=["raw_image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),
                #ScaleIntensityd(keys=cond_keys, minv=-1, maxv=1), #如果是label数据需要删掉这句
                
                #concat(CT,Mask)
                ConcatenateImageAndMask(keys=["raw_image", "mask"], concat_key='image', dim=0),
            ]
        )
    else:
        train_transforms = Compose(
            [
                LoadImaged(keys=load_keys, ensure_channel_first=True),
                # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resized(keys=["raw_image"], spatial_size=resize_size),
                Resized(keys=["mask"], spatial_size=resize_size),
                ScaleIntensityRanged(keys=["raw_image"], a_min=-1000, a_max=400, b_min=-1, b_max=1, clip=True),  # 如果是label数据需要删掉这句
            
                #concat(CT,Mask)
                ConcatenateImageAndMask(keys=["raw_image", "mask"], concat_key='image', dim=0),
            ]
        )
    # print(f"data_dicts:{data_dicts}")
    if split == "train":
        train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, num_workers=8)
    else:
        train_ds = Dataset(data=data_dicts, transform=train_transforms)
    shuffle = False if split == "test" or split == "val" else True
    #
    #datas = [train_ds[i] for i in range(len(train_ds))]  # 6min30s
    #M 5min30s 8min30s
    #D 4min30s
    train_dl = MultiEpochsDataLoader(train_ds, batch_size=bs, pin_memory=True, num_workers=4, drop_last=True, shuffle=shuffle)
    return train_dl


@hydra.main(config_path="../conf", config_name="config/autoencoder.yaml", version_base="1.3")
def main(config):
    config = config["config"]
    # train_dl = prepare_dataset(
    #     data_path=config.data_path, resize_size=config.resize_size, cond_path=config.cond_path, split="test"
    # )
    test_dl = prepare_dataset(
        cond_path="/disk/syz/Xray-Diffsuion/datacsv/TotalT2Idata.csv",
        #cond_path=None,
        data_path="/disk/syz/Xray-Diffsuion/datacsv/TotalT2Idata.csv",
        resize_size=config.resize_size,
        img_resize_size=config.resize_size,
        split="train",
    )
    print(f"test_dl_len:{len(test_dl)}")
    test_dl = prepare_dataset(
        cond_path="/disk/syz/Xray-Diffsuion/datacsv/TotalT2Idata.csv",
        #cond_path=None,
        data_path="/disk/syz/Xray-Diffsuion/datacsv/TotalT2Idata.csv",
        resize_size=config.resize_size,
        img_resize_size=config.resize_size,
        split="test",
    )
    print(f"train_dl_len:{len(test_dl)}")
    
    test_dl = prepare_dataset(
        cond_path="/disk/syz/Xray-Diffsuion/datacsv/TotalT2Idata.csv",
        #cond_path=None,
        data_path="/disk/syz/Xray-Diffsuion/datacsv/TotalT2Idata.csv",
        resize_size=config.resize_size,
        img_resize_size=config.resize_size,
        split="val",
    )
    print(f"val_dl_len:{len(test_dl)}")
    
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
        img = i["raw_image"]
        print(f"img.shape:{img.shape}")
        # img = img.squeeze(0)
        # img = img * 255
        mask = i["mask"]
        print(f"mask.shape:{mask.shape}")
        
        c_image = i["image"]
        print(f"c_image.shape:{c_image.shape}")   # (1, 2, 128, 128, 128)
        # 分割通道
        channel1 = c_image[:, 0, :, :, :].squeeze(0)  # 形状为 (128, 128, 128)
        channel2 = c_image[:, 1, :, :, :].squeeze(0)  # 形状为 (128, 128, 128)
        channel1 = (channel1 + 1) * 127.5
        img = (img + 1) * 127.5        
        img = img.reshape(1, 128, 128, 128)
        mask = mask.reshape(1, 128, 128, 128)
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
            output_postfix="mask",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        saver_cat1(mask)
        saver_cat2 = SaveImage(
            output_dir="./",
            output_ext=".nii.gz",
            output_postfix="channel1",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        saver_cat2(channel1)
        saver_cat3 = SaveImage(
            output_dir="./",
            output_ext=".nii.gz",
            output_postfix="channel2",
            separate_folder=False,
            output_dtype=np.uint8,
            # scale=255,
            resample=False,
            squeeze_end_dims=True,
            writer="NibabelWriter",
        )
        saver_cat3(channel2)
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
