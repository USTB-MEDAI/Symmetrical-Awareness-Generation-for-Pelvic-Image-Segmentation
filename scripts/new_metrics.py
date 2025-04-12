from pathlib import Path
import SimpleITK as sitk
import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ldm.util import AverageMeter


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    L2_distance = torch.cdist(total, total) ** 2
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)    

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)



def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


if __name__ == "__main__":
    # data_path = "/disk/cyq/2024/My_Proj/VQGAN-DDPM/logs/cddpm/pl_test_cddpm-2024-06-24/15-07-57"
    # data_path = "/disk/cyq/2024/My_Proj/VQGAN-DDPM/logs/ldm/pl_test_ldm-2024-06-14/10-01-33"
    # data_path = "/disk/cyq/2024/My_Proj/VQGAN-DDPM/logs/c_vqgan_transformer/pl_test_transformer-2024-07-03/20-51-05"
    # data_path = "/home/syz/Xray-Diffsuion/logs/autoencoder/pl_test_autoencoder-2024-12-23/17-43-05"  #label ae CT图像
    # data_path = "/home/syz/Xray-Diffsuion/logs/autoencoder/pl_test_autoencoder-2025-01-07/00-50-13"  #label ae 标签图像 1000
    # data_path = "/home/syz/Xray-Diffsuion/logs/autoencoder/pl_test_autoencoder-2024-12-25/00-20-25"  #label ae 标签图像 190
    
    # data_path = "/home/syz/Xray-Diffsuion/logs/ldm/pl_test_ldm-2024-12-30/14-29-07"  #ldm 标签cond 1000
    # data_path = "/home/syz/Xray-Diffsuion/logs/ldm/pl_test_ldm-2025-01-10/11-18-34"  #ldm 标签cond 650
    # data_path = "/home/syz/Xray-Diffsuion/logs/autoencoder/pl_test_autoencoder-2025-01-08/02-27-31"  #label ae 标签图像 1000
    
    #data_path = "/home/syz/Xray-Diffsuion/logs/ldm/pl_test_ldm-2025-01-14/10-18-12"  # ldm 标签cond 940 ckpt_nontrainable
    data_path = "/disk/SYZ/Xray-Diffsuion/logs/ldm_T2I/pl_test_ldm-2025-04-01/23-12-47"  # ldm 标签cond 1000 ckpt_trainable
    # real_path = "/nvme/Pelvic1k/Left_hip2/source_cropped"
    # rec_path = "/nvme/Pelvic1k/Right_hip3/source_cropped"
    
    psnr_record_pl = AverageMeter()
    ssim_record_pl = AverageMeter()
    fid_record_pl = AverageMeter()
    lpip_record_pl = AverageMeter()

    psnr_pl = PeakSignalNoiseRatio()
    ssim_pl = StructuralSimilarityIndexMeasure(data_range=4095)
    lpip_pl = LearnedPerceptualImagePatchSimilarity()
    fid_pl = FrechetInceptionDistance(feature=2048)

    # ori_mhd_list = sorted(Path(data_path).glob("*origin*.nii"))
    # recon_mhd_list = sorted(Path(data_path).glob("*reconstructions*.nii"))
    # ori_mhd_list = sorted(Path(data_path).glob("*origin*.mhd"))
    # recon_mhd_list = sorted(Path(data_path).glob("*reconstructions*.mhd"))

    # ori_mhd_list = sorted(Path(data_path).glob("*label_x*.nii.gz"))
    # recon_mhd_list = sorted(Path(data_path).glob("*label_rec*.nii.gz"))
    
    # ori_mhd_list = sorted(Path(data_path).glob("*origin_x*.nii.gz"))
    # recon_mhd_list = sorted(Path(data_path).glob("*ae_rec*.nii.gz"))
    
    # ori_mhd_list = sorted(Path(real_path).glob("*.nii.gz"))
    # recon_mhd_list = sorted(Path(rec_path).glob("*.nii.gz"))
    #将加载的输入都重采样为128*128*128
    # for i in range(len(ori_mhd_list)):
    #     ori_img = sitk.ReadImage(str(ori_mhd_list[i]))
    #     ori_arr = sitk.GetArrayFromImage(ori_img)
    #     ori_arr = torch.tensor(ori_arr).to(torch.float32)
    #     ori_arr = ori_arr[None, None,::]
    #     ori_arr = torch.nn.functional.interpolate(ori_arr, size=(128, 128, 128), mode='trilinear', align_corners=False)
    #     sitk.WriteImage(sitk.GetImageFromArray(ori_arr.numpy()), str(ori_mhd_list[i]))
    #     print(f"ori_arr shape: {ori_arr.shape}")
    
    # for i in range(len(recon_mhd_list)):
    #     recon_img = sitk.ReadImage(str(recon_mhd_list[i]))
    #     recon_arr = sitk.GetArrayFromImage(recon_img)
    #     recon_arr = torch.tensor(recon_arr).to(torch.float32)
    #     recon_arr = recon_arr[None, None,::]
    #     recon_arr = torch.nn.functional.interpolate(recon_arr, size=(128, 128, 128), mode='trilinear', align_corners=False)
    #     sitk.WriteImage(sitk.GetImageFromArray(recon_arr.numpy()), str(recon_mhd_list[i]))
    #     print(f"recon_arr shape: {recon_arr.shape}")
    
    # ori_mhd_list = sorted(Path(data_path).glob("*label_x*.nii.gz"))
    # # 使用列表解析排除以 'ae_rec' 结尾的文件
    
    ori_mhd_list = sorted(Path(data_path).glob("*origin_x*.nii.gz"))
    # 使用列表解析排除以 'ae_rec' 结尾的文件
    recon_mhd_list = sorted([f for f in Path(data_path).glob("*rec*.nii.gz") if not f.name.endswith("ae_rec.nii.gz")])
    recon_mhd_list = sorted('label' not in str(f) for f in recon_mhd_list)

    print(recon_mhd_list)
    
    whole_recon = []
    whole_ori = []
    for ori, recon in tqdm.tqdm(zip(ori_mhd_list, recon_mhd_list), total=len(ori_mhd_list)):
        ori_img = sitk.ReadImage(str(ori))
        ori_arr = sitk.GetArrayFromImage(ori_img)
        ori_arr = torch.tensor(ori_arr).to(torch.float32)
        ori_arr = ori_arr[None, None,::]


        recon_img = sitk.ReadImage(str(recon))
        recon_arr = sitk.GetArrayFromImage(recon_img)
        recon_arr = torch.tensor(recon_arr).to(torch.float32)
        recon_arr = recon_arr[None, None,::]


        # whole_recon = recon_arr if len(whole_recon) == 0 else torch.cat((whole_recon, recon_arr), dim=0)
        # whole_ori = ori_arr if len(whole_ori) == 0 else torch.cat((whole_ori, ori_arr), dim=0)

        psnr = psnr_pl(recon_arr, ori_arr)
        ssim = ssim_pl(recon_arr, ori_arr)
        
        slices = recon_arr.shape[2]
        
        for i in range(slices):
            # 提取第i个切片，并调整形状为(1, 1, 128, 128)
            slice1 = recon_arr[:, :, i, :, :].squeeze(0)  # (1, 128, 128)
            slice2 = ori_arr[:, :, i, :, :].squeeze(0)
            
            # 归一化到[-1, 1]范围
            # slice1 = slice1 / 127.5 - 1
            # slice2 = slice2 / 127.5 - 1
            
            #归一化到[-1, 1]范围
            min_val = min(slice1)
            max_val = max(slice1)
            slice1 = [2 * ((x - min_val) / (max_val - min_val)) - 1 for x in slice1]
            slice1 = torch.tensor(slice1)
            
            min_val = min(slice2)
            max_val = max(slice2)
            slice2 = [2 * ((x - min_val) / (max_val - min_val)) - 1 for x in slice2]
            slice2 = torch.tensor(slice2)
            
            # 添加通道维度，复制通道变为3通道，调整形状为(1, 3, 128, 128)
            slice1 = slice1.unsqueeze(0).repeat(1, 3, 1, 1)
            slice2 = slice2.unsqueeze(0).repeat(1, 3, 1, 1)
            # print(f"slice1 shape: {slice1.shape}, slice2 shape: {slice2.shape}")
            #计算LPIPS分数
            lpip = lpip_pl(slice1, slice2)
            lpip_record_pl.update(lpip)
            
            #计算FID分数
            fid_pl.update(torch.tensor(slice1).to(torch.uint8), real = False)
            fid_pl.update(torch.tensor(slice2).to(torch.uint8), real = True)
            
        recon_arr_unit8 = torch.tensor(recon_arr).to(torch.uint8)
        ori_arr_unit8 = torch.tensor(ori_arr).to(torch.uint8)
        ori_arr_unit8 = ori_arr_unit8.squeeze(0)
        recon_arr_unit8 = recon_arr_unit8.squeeze(0)
        # print(f"recon_arr_unit8 shape: {recon_arr_unit8.shape}, ori_arr_unit8 shape: {ori_arr_unit8.shape}")
        # fid_pl.update(recon_arr_unit8, real=False)
        # fid_pl.update(ori_arr_unit8, real=True)

        psnr_record_pl.update(psnr)
        ssim_record_pl.update(ssim)
    
    fid_score = fid_pl.compute()

    # whole_ori = whole_ori.permute(1, 0, 2, 3, 4).view(whole_ori.shape[0], -1)
    # whole_recon = whole_recon.permute(1, 0, 2, 3, 4).view(whole_recon.shape[0], -1)
    # mmd_score = mmd(whole_recon, whole_ori)

    print(f"PSNR mean±std:{psnr_record_pl.mean}±{psnr_record_pl.std}")
    print(f"SSIM mean±std:{ssim_record_pl.mean}±{ssim_record_pl.std}")
    print(f"LPIPS mean±std:{lpip_record_pl.mean}±{lpip_record_pl.std}")
    # # # print(f"FID mean±std:{fid_record_pl.mean}±{fid_record_pl.std}")
    print(f"FID mean±std:{fid_score}")
    # print(f"MMD:{mmd_score}")
