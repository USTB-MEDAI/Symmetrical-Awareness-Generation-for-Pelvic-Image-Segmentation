import torch
# import pytorch_lightning as pl
import lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from contextlib import contextmanager
import hydra
import numpy as np
#import SimpleITK as sitk
from monai.transforms import SaveImage

#from .quantize import VectorQuantizer2 as VectorQuantizer
from .model import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution

from .discriminator import LPIPSWithDiscriminator
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
from tqdm import tqdm
from ..fp16_util import convert_module_to_f16, convert_module_to_f32


class VQModelInterface:
    def __init__(self) -> None:
        pass


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        sync_dist=False,
        save_interval=50,
        save_path=None,
        base_learning_rate=None,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
    ):
        super().__init__()
        # * manual optimization
        self.z_sample_ls = []
        self.z_mean_ls = []
        self.automatic_optimization = False
        self.save_interval = save_interval
        self.root_path = save_path
        self.sync_dist = sync_dist

        self.learning_rate = base_learning_rate
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        if lossconfig:
            self.loss = LPIPSWithDiscriminator(**lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            print("!!!!!!!!!!!!")
            print(f"Loading model from {ckpt_path}")
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            pass
        # self.convert_to_fp16()

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.encoder.apply(convert_module_to_f16)
        self.decoder.apply(convert_module_to_f16)
        self.loss.apply(convert_module_to_f16)
        self.quant_conv.apply(convert_module_to_f16)
        self.post_quant_conv.apply(convert_module_to_f16)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # 对z进行镜像翻转
        # print(f"z shape :{z.shape}")
        # z = torch.flip(z, dims=[2])
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()  # 释放显存
        opt_ae, opt_disc = self.optimizers()

        inputs = batch["image"]
        reconstructions, posterior = self(inputs)
        # print(f"inputs shape :{inputs.shape}")
        # print(f"reconstructions shape :{reconstructions.shape}")
        # print(f"posterior shape :{posterior.mean.shape}")
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # train the discriminator
        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.sync_dist)
        
        

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()  # 释放显存

        if self.current_epoch % 10 == 0:
            inputs = batch["image"]
            reconstructions, posterior = self(inputs)

            # reconstructions = torch.clamp(reconstructions, min=-1, max=1)
            # reconstructions = (reconstructions + 1) * 127.5

            # inputs = torch.clamp(inputs, min=-1, max=1)
            # inputs = (inputs + 1) * 127.5

            rec_loss = F.mse_loss(reconstructions, inputs)

            # reconstructions = reconstructions.squeeze(0).permute(1, 0, 2, 3)
            # reconstructions = reconstructions.type(torch.uint8)
            # grid = torchvision.utils.make_grid(reconstructions)
            # self.logger.experiment.add_image("val_images", grid, self.global_step)

            # inputs = inputs.type(torch.uint8)
            # inputs = inputs.squeeze(0).permute(1, 0, 2, 3)
            # inputs = inputs.type(torch.uint8)
            # grid = torchvision.utils.make_grid(inputs)
            # self.logger.experiment.add_image("val_inputs", grid, self.global_step)

            self.log("val/rec_loss", rec_loss, sync_dist=self.sync_dist)

    def img_saver(self, img, post_fix, i_type=".nii", meta_data=None, random_num=None,**kwargs):
        """
        save img to self.root_path with post_fix

        Args:
            img (torch.Tensor): [description]
            post_fix (str): [description]
            type (str, optional): [description]. Defaults to "nii".
            meta_data ([type], optional): [description]. Defaults to None.
        """
        if hasattr(img, "meta"):
            meta_data = img.meta
        else:
            print("img dosen't has meta attribution use `None` as meta_dat")

        assert i_type in [".nii", ".nii.gz", ".jpg"], "Only .nii or .jpg suffix file supported now"
        assert post_fix in ["origin_x", "ae_rec", "label_x", "label_rec","z_sample","xray1", "xray2", "rec"], "unsupported post_fix"

        img = img.squeeze(0)
        print(f"max value :{torch.max(img)}")
        print(f"min value :{torch.min(img)}")
        
        writer = "NibabelWriter" if "nii" in i_type else "PILWriter"
        out_ext = ".nii.gz" if "nii" in i_type else ".jpg"
        
        out_ext = f"{random_num}{out_ext}"
        # if post_fix == "ae_rec":
        #     MAX = torch.max(img)
        #     MIN = torch.min(img)
        #     img = 2*(img-MAX)/(MAX-MIN)-1
        # else:
        if post_fix == "label_x":
            #img = torch.clamp(img, min=0, max=1)
            print(f"max value :{torch.max(img)}")
            print(f"min value :{torch.min(img)}")
            saver = SaveImage(
                output_dir=self.root_path,
                output_ext=out_ext,
                output_postfix=post_fix,
                separate_folder=False,
                output_dtype=np.uint8,
                resample=False,
                squeeze_end_dims=True,
                writer=writer,
                **kwargs,
                )
            saver(img, meta_data=meta_data)
            
        elif post_fix == "label_rec":
            # 将图像数据中不为0的值变为1
            
            img = torch.clamp(img, min=-1, max=1) 
            img_cpu = img.cpu().numpy()
            #img = (img + 1) * 0.5
            # with open(os.path.join(self.root_path, "label_rec.txt"), "a") as f:
            #      f.write(str(img_cpu))
            #      f.write("\n")
            img = torch.where(img > 0.6, 1, 0)
            img = img.to(torch.uint8)
            
            #将img值打印到txt文件中
            

            print(f"max value :{torch.max(img)}")
            print(f"min value :{torch.min(img)}")
            #img = (img == 255) | (img == 254)
            print(f"max value :{torch.max(img)}")
            print(f"min value :{torch.min(img)}")
            saver = SaveImage(
                output_dir=self.root_path,
                output_ext=out_ext,
                output_postfix=post_fix,
                separate_folder=False,
                output_dtype=np.uint8,
                resample=False,
                squeeze_end_dims=True,
                writer=writer,
                **kwargs,
                )
            saver(img, meta_data=meta_data)
            
        else:
            img = torch.clamp(img, min=-1, max=1)
            img = (img + 1) * 127.5
            saver = SaveImage(
                output_dir=self.root_path,
                output_ext=out_ext,
                output_postfix=post_fix,
                separate_folder=False,
                output_dtype=np.uint8,
                resample=False,
                squeeze_end_dims=True,
                writer=writer,
                **kwargs,
            )
            saver(img, meta_data=meta_data)

    def test_step(self, batch, batch_idx):
        import random
        random_num = random.sample(range(1, 100), 1)
        
        inputs = batch["image"]
        channel1 = inputs[:, 0, :, :, :]
        channel2 = inputs[:, 1, :, :, :]
        
        reconstructions, posterior = self(inputs)
        reconstructions = reconstructions.detach().to(dtype=torch.float32) 
        res_channel1 = reconstructions[:, 0, :, :, :]
        res_channel2 = reconstructions[:, 1, :, :, :]
        
        self.img_saver(channel1, post_fix="origin_x", random_num=random_num)
        self.img_saver(channel2, post_fix="label_x", random_num=random_num)
        
        self.img_saver(res_channel1, post_fix="ae_rec", random_num=random_num)
        self.img_saver(res_channel2, post_fix="label_rec", random_num=random_num)

        # self.img_saver(inputs, post_fix="label_x")
        # self.img_saver(reconstructions, post_fix="label_rec")     

        # self.img_saver(inputs, post_fix="origin_x")
        # self.img_saver(reconstructions, post_fix="ae_rec")

        # image = sitk.GetImageFromArray(reconstructions)
        # sitk.WriteImage(image, os.path.join(self.save_path, f"reconstructions_{batch_idx}.mhd"))

        # inputs = to_image(inputs)
        # image = sitk.GetImageFromArray(inputs)
        # sitk.WriteImage(image, os.path.join(self.save_path, f"origin_{batch_idx}.mhd"))
        # save_path = os.path.join(self.save_path, "val_reconstruction")
        # os.makedirs(save_path, exist_ok=True)
        # image = sitk.GetImageFromArray(reconstructions)
        # sitk.WriteImage(image, os.path.join(save_path, f"{self.global_step}.mhd"))

        # self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=self.sync_dist)
        # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.sync_dist)
        # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=self.sync_dist)
        # return self.log_dic

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def to_image(x):
        x = torch.clamp(x, min=-1, max=1)
        x = (x + 1) * 127.5
        # x = x.squeeze(0).permute(1, 0, 2, 3)
        x = x.type(torch.uint8)
        x = x.cpu().numpy()
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


@hydra.main(version_base=None, config_path="/home/syz/Xray-Diffsuion/conf", config_name="/config/autoencoder.yaml")
def main(config):
    config = config["config"]
    #model_config = config["model"]
    # ddconfig = config["model"]["params"]["ddconfig"]
    # lossconfig = config["model"]["params"]["lossconfig"]
    # print(model_config.get("params", dict()))
    model = AutoencoderKL(**config["model"])
    #input = torch.randn((1, 1, 16, 256, 256))
    input = torch.randn((1, 2, 128, 128, 128))
    output, posterior = model(input)
    print(f"output.shape: {output.shape}")
    
    # Assuming 'distribution' is an instance of DiagonalGaussianDistribution
    mean = posterior.mean
    std = posterior.std

    # print(f"mean: {mean}")
    # print(f"std: {std}")


if __name__ == "__main__":
    main()
