from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHook:
    def __init__(self):
        self.attention_maps = []

    def hook_fn(self, module, input, output):
        # 对于PyTorch的nn.MultiheadAttention
        # output格式: (attn_output, attn_weights_optional)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1].detach().cpu()  # 获取注意力权重
            self.attention_maps.append(attn_weights)
        else:
            # 对于自定义注意力实现
            attn = output.detach().cpu()
            self.attention_maps.append(attn)
            
def overlay_2d_slice(volume, attn, slice_idx=64):
    """在三个解剖平面显示叠加效果"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # 轴向切片
    axes[0, 0].imshow(volume[slice_idx], cmap='gray')
    axes[0, 0].set_title("Original (Axial)")
    axes[0, 1].imshow(attn[slice_idx], cmap='hot')
    axes[0, 2].imshow(volume[slice_idx], cmap='gray')
    axes[0, 2].imshow(attn[slice_idx], cmap='hot', alpha=0.5)
    axes[0, 2].set_title("Overlay")
    
    # 冠状切片
    axes[1, 0].imshow(volume[:, slice_idx], cmap='gray')
    axes[1, 1].imshow(attn[:, slice_idx], cmap='hot')
    axes[1, 2].imshow(volume[:, slice_idx], cmap='gray')
    axes[1, 2].imshow(attn[:, slice_idx], cmap='hot', alpha=0.5)
    
    # 矢状切片
    axes[2, 0].imshow(volume[:, :, slice_idx], cmap='gray')
    axes[2, 1].imshow(attn[:, :, slice_idx], cmap='hot')
    axes[2, 2].imshow(volume[:, :, slice_idx], cmap='gray')
    axes[2, 2].imshow(attn[:, :, slice_idx], cmap='hot', alpha=0.5)
    
    plt.tight_layout()
    return fig

import plotly.graph_objects as go
from ipywidgets import interactive

def interactive_3d_overlay(volume, attn, threshold=0.3):
    """交互式三维可视化"""
    # 创建原始体积的等值面
    vol_mesh = go.Isosurface(
        x=np.arange(128).flatten(),
        y=np.arange(128).flatten(),
        z=np.arange(128).flatten(),
        value=volume.flatten(),
        isomin=0.3,
        isomax=0.7,
        surface_count=1,
        colorscale='gray',
        opacity=0.3
    )
    
    # 创建注意力区域的等值面
    attn_mesh = go.Isosurface(
        x=np.arange(128).flatten(),
        y=np.arange(128).flatten(),
        z=np.arange(128).flatten(),
        value=attn.flatten(),
        isomin=threshold,
        isomax=1.0,
        surface_count=5,
        colorscale='hot',
        opacity=0.5
    )
    
    fig = go.Figure(data=[vol_mesh, attn_mesh])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(t=0, b=0, l=0, r=0)
    )
    return fig

# 添加阈值调节控件
def update_threshold(threshold=(0.1, 1.0, 0.05)):
    return interactive_3d_overlay(volume, attn, threshold=threshold)

interactive(update_threshold)