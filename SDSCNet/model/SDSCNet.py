import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from mmengine.model import constant_init
from einops import rearrange
import typing as t

from typing import List, Callable
from torch import Tensor

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,activation='relu'):
        super(BasicConv2d, self).__init__()
        self.activation=activation
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class PDC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(PDC, self).__init__()
        
        self.LDC_5  = DCNConv(inc=in_ch, outc=out_ch, num_param=5)
        self.bn_5   = nn.BatchNorm2d(out_ch)
        self.silu_5 = nn.SiLU()

        self.LDC_9  = DCNConv(inc=in_ch, outc=out_ch, num_param=9)
        self.bn_9   = nn.BatchNorm2d(out_ch)
        self.silu_9 = nn.SiLU()

        self.LDC_13  = DCNConv(inc=in_ch, outc=out_ch, num_param=13)
        self.bn_13   = nn.BatchNorm2d(out_ch)
        self.silu_13 = nn.SiLU()

        self.fuse_conv = nn.Conv2d(3 * out_ch, out_ch, kernel_size=1, bias=False)
        self.fuse_bn   = nn.BatchNorm2d(out_ch)
        self.fuse_act  = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.silu_5 (self.bn_5 (self.LDC_5 (x)))  
        x2 = self.silu_9 (self.bn_9 (self.LDC_9 (x)))
        x3 = self.silu_13(self.bn_13(self.LDC_13(x)))

        y = torch.cat([x1, x2, x3], dim=1)              # B, 3*out_ch, H, W
        y = self.fuse_act(self.fuse_bn(self.fuse_conv(y)))
        return x + y


class DCNConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=False):
        super(DCNConv, self).__init__()
        self.num_param = num_param
        self.stride = stride

        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=False),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )

        self.p_conv = nn.Conv2d(inc, 3 * num_param, kernel_size=3, padding=1, stride=stride, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.p_conv.weight)  
            nn.init.zeros_(self.p_conv.bias)    

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        
        return

    def forward(self, x):
        B, C, H, W = x.shape
        off_mask = self.p_conv(x)                               # (B, 3N, H, W)
        N = self.num_param
        offset, mask_logits = torch.split(off_mask, [2*N, N], dim=1)
        mask = torch.sigmoid(mask_logits).permute(0, 2, 3, 1)   # (B, H, W, N)

        dtype = offset.data.type()
        # (B, 2N, H, W)
        p = self._get_p(offset, dtype)

        # (B, H, W, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [torch.clamp(q_lt[..., :N], 0, H - 1), torch.clamp(q_lt[..., N:], 0, W - 1)],
            dim=-1
        ).long()
        q_rb = torch.cat(
            [torch.clamp(q_rb[..., :N], 0, H - 1), torch.clamp(q_rb[..., N:], 0, W - 1)],
            dim=-1
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat(
            [torch.clamp(p[..., :N], 0, H - 1), torch.clamp(p[..., N:], 0, W - 1)],
            dim=-1
        )

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)   # (B, C, H, W, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

       
        x_offset = (g_lt.unsqueeze(1) * x_q_lt
                    + g_rb.unsqueeze(1) * x_q_rb
                    + g_lb.unsqueeze(1) * x_q_lb
                    + g_rt.unsqueeze(1) * x_q_rt)

        x_offset = x_offset * mask.unsqueeze(1)                 # (B, C, H, W, N)

        x_offset = self._reshape_x_offset(x_offset, N)          # (B, C, H*N, W)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        device = self.p_conv.weight.device

        if self.num_param == 5:
            half = N // 2  # 2
            dx = torch.arange(-half, half + 1, device=device, dtype=torch.float32)  
            dy = torch.zeros_like(dx, device=device, dtype=torch.float32)           
            return torch.cat([dx, dy], dim=0).view(1, 2*N, 1, 1).type(dtype)

        if self.num_param == 9:
            coords = [(-1,-1), (-1, 0), (-1, 1),
                      ( 0,-1), ( 0, 0), ( 0, 1),
                      ( 1,-1), ( 1, 0), ( 1, 1)]
            xs = torch.tensor([x for x, _ in coords], device=device, dtype=torch.float32)
            ys = torch.tensor([y for _, y in coords], device=device, dtype=torch.float32)
            return torch.cat([xs, ys], dim=0).view(1, 2*N, 1, 1).type(dtype)

        if self.num_param == 13:
            coords = [( 1,0), (-1,0), (0, 1), (0,-1),
                      ( 2,0), (-2,0), (0, 2), (0,-2),
                      ( 1,1), ( 1,-1), (-1,1), (-1,-1),
                      ( 0,0)]
            xs = torch.tensor([x for x, _ in coords], device=device, dtype=torch.float32)
            ys = torch.tensor([y for _, y in coords], device=device, dtype=torch.float32)
            return torch.cat([xs, ys], dim=0).view(1, 2*N, 1, 1).type(dtype)

        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number, device=device),
            torch.arange(0, base_int, device=device),
            indexing='ij'
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1, device=device),
                torch.arange(0, mod_number, device=device),
                indexing='ij'
            )
            p_n_x = torch.cat((p_n_x, torch.flatten(mod_p_n_x)))
            p_n_y = torch.cat((p_n_y, torch.flatten(mod_p_n_y)))
        return torch.cat([p_n_x, p_n_y], 0).view(1, 2*N, 1, 1).type(dtype)

    def _get_p_0(self, h, w, N, dtype):
        device = self.p_conv.weight.device
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride, device=device),
            torch.arange(0, w * self.stride, self.stride, device=device),
            indexing='ij'
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)   # (1, 2N, 1, 1)
        p_0 = self._get_p_0(h, w, N, dtype)  # (1, 2N, h, w)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)  # (b, c, h*w)

        index = q[..., :N] * padded_w + q[..., N:]           # (b, h, w, N)
        index = index.contiguous().unsqueeze(1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


class ECC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ECC, self).__init__()
        c_mid = max(8, int(out_ch * 0.5))  # mid_ratio = 0.5

        self.conv1  = BasicConv2d(
            in_planes=in_ch, out_planes=out_ch,
            kernel_size=3, padding=1, dilation=1, activation='silu'
        )

        self.dconv1 = BasicConv2d(
            in_planes=out_ch, out_planes=c_mid,
            kernel_size=3, padding=2, dilation=2, activation='silu'
        )
        self.dconv2 = BasicConv2d(
            in_planes=c_mid, out_planes=c_mid,
            kernel_size=3, padding=3, dilation=3, activation='silu'
        )

        self.dconv3 = BasicConv2d(
            in_planes=c_mid*2, out_planes=out_ch,
            kernel_size=3, padding=2, dilation=2, activation='silu'
        )

        self.conv2  = BasicConv2d(
            in_planes=out_ch*2, out_planes=out_ch,
            kernel_size=3, padding=1, dilation=1, activation='silu'
        )

        self.proj = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))

    def forward(self, x):
        x_res = self.proj(x)

        x1  = self.conv1(x)                              # B, out_ch, H, W
        dx1 = self.dconv1(x1)                            # B, c_mid, H, W
        dx2 = self.dconv2(dx1)                           # B, c_mid, H, W
        dx3 = self.dconv3(torch.cat([dx1, dx2], dim=1))  # B, out_ch, H, W
        out = self.conv2(torch.cat([x1, dx3], dim=1))    # B, out_ch, H, W

        return out + x_res
 
# SAAM
class SAAM(nn.Module):
    def __init__(self, in_ch, out_ch,spatial_kernel_sizes=[3, 5, 7, 9],channel_kernel_sizes=[7, 11, 21],heads = [1,8]):
        super(SAAM,self).__init__()
        self.lrssa = LRSSA(in_ch,spatial_kernel_sizes)
        self.mscsa = MSCSA(in_ch,channel_kernel_sizes,heads)
        self.mscb = MSCB(in_ch, out_ch)
      
    def forward(self, x):
        x = self.lrssa(x)
        x = self.mscsa(x)
        x = self.mscb(x)
        return x
    
class LRSSA(nn.Module):
    def __init__(self, in_ch,spatial_kernel_sizes):
        super(LRSSA,self).__init__()
        self.in_ch=in_ch 
        self.dynamic_convs = nn.ModuleList([
                DynamicConv1d(in_ch // 4, in_ch // 4, spatial_kernel_sizes)
                for _ in range(4)  
        ])
        self.hnorm = nn.GroupNorm(4, in_ch)
        self.wnorm = nn.GroupNorm(4, in_ch)
        self.sig = nn.Sigmoid()

      
    def forward(self, x):
        b, c, h, w = x.size()
        hx = x.mean(dim=3)  # [B,C,H]
        wx = x.mean(dim=2)  # [B,C,W]
        
        hx_parts = torch.split(hx, self.in_ch//4, dim=1)
        wx_parts = torch.split(wx, self.in_ch//4, dim=1)
        
        hx_out = []
        wx_out = []

        for i in range(4):
            hx_out.append(self.dynamic_convs[i](hx_parts[i]))  
            wx_out.append(self.dynamic_convs[i](wx_parts[i]))
        hx_attn = self.sig(self.hnorm(torch.cat(hx_out, dim=1)))
        wx_attn = self.sig(self.wnorm(torch.cat(wx_out, dim=1)))
        hx_attn = hx_attn.view(b, c, h, 1)
        wx_attn = wx_attn.view(b, c, 1, w)
        return x * hx_attn * wx_attn
    
class DynamicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.experts = len(kernel_sizes)
        self.conv_kernels = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, ks, 
                      padding=ks//2, groups=in_channels)
            for ks in kernel_sizes
        ])
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, self.experts, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # x: [B, C, L]
        attn_weights = self.attn(x)  # [B, experts]
        outputs = 0.0
        for i in range(self.experts):
            w = attn_weights[:, i].view(-1, 1, 1)  # [B,1,1]
            outputs += w * self.conv_kernels[i](x)
        return outputs

class MSCSA(nn.Module):
    def __init__(self, in_ch,channel_kernel_sizes=[7, 11, 21],heads = [1,8]):
        super(MSCSA,self).__init__()
        self.lrb = SHSA(in_ch, channel_kernel_sizes, heads[0], in_ch // heads[0])
        self.srb = MHSA(in_ch, channel_kernel_sizes, heads[1], in_ch // heads[1])
        self.weights = nn.Parameter(torch.ones(4))  
        self.sig = nn.Sigmoid()
   
    def forward(self, x):
        lrb_h, lrb_w= self.lrb(x)
        srb_h, srb_w= self.srb(x)
        attn = (self.weights[0] * lrb_h + self.weights[1] * lrb_w + self.weights[2] * srb_h + self.weights[3] * srb_w)
        attn = self.sig(attn)
        return attn * x

class SHSA(nn.Module):
    def __init__(self, in_ch, channel_kernel_sizes,heads, head_dim):
        super(SHSA,self).__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scaler=self.head_dim ** -0.5 
        self.q_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[0],1), padding=(3, 0),bias=False, groups=in_ch)
        self.k_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[1],1), padding=(5, 0),bias=False, groups=in_ch)
        self.v_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[2],1), padding=(10, 0),bias=False, groups=in_ch)
        self.q_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[0]), padding=(0, 3),bias=False, groups=in_ch)
        self.k_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[1]), padding=(0, 5),bias=False, groups=in_ch)
        self.v_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[2]), padding=(0, 10),bias=False, groups=in_ch)
      
    def forward(self, x):
        _, _, h, w = x.size()
        q_h = self.q_h(x)
        k_h = self.k_h(x)
        v_h = self.v_h(x)
        q_w = self.q_w(x)
        k_w = self.k_w(x)
        v_w = self.v_w(x)

        q_h = rearrange(q_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_h = rearrange(k_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_h = rearrange(v_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)

        attn_h = q_h @ k_h.transpose(-2, -1)
        attn_h = attn_h * self.scaler
        attn_h = attn_h.softmax(dim=-1)
        attn_h = attn_h @ v_h
        attn_h = rearrange(attn_h, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_h = attn_h.mean((2, 3), keepdim=True)

        q_w = rearrange(q_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_w = rearrange(k_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_w = rearrange(v_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        
        attn_w = q_w @ k_w.transpose(-2, -1)
        attn_w = attn_w * self.scaler 
        attn_w = attn_w.softmax(dim=-1)
    
        attn_w = attn_w @ v_w
        attn_w = rearrange(attn_w, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_w = attn_w.mean((2, 3), keepdim=True)

        return attn_h,attn_w
    
class MHSA(nn.Module):
    def __init__(self, in_ch, channel_kernel_sizes, heads, head_dim):
        super(MHSA,self).__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scaler=self.head_dim ** -0.5 
        self.q_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[0],1), padding=(3, 0),bias=False, groups=in_ch)
        self.k_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[1],1), padding=(5, 0),bias=False, groups=in_ch)
        self.v_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[2],1), padding=(10, 0),bias=False, groups=in_ch)
        self.q_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[0]), padding=(0, 3),bias=False, groups=in_ch)
        self.k_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[1]), padding=(0, 5),bias=False, groups=in_ch)
        self.v_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[2]), padding=(0, 10),bias=False, groups=in_ch)
    
    def forward(self, x):
        _, _, h, w = x.size()

        q_h = self.q_h(x)
        k_h = self.k_h(x)
        v_h = self.v_h(x)
        q_w = self.q_w(x)
        k_w = self.k_w(x)
        v_w = self.v_w(x)

        q_h = rearrange(q_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_h = rearrange(k_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_h = rearrange(v_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)

        attn_h = q_h @ k_h.transpose(-2, -1)
        attn_h = attn_h * self.scaler
        attn_h = attn_h.softmax(dim=-1)
        attn_h = attn_h @ v_h
        attn_h = rearrange(attn_h, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_h = attn_h.mean((2, 3), keepdim=True)

        q_w = rearrange(q_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_w = rearrange(k_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_w = rearrange(v_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        
        attn_w = q_w @ k_w.transpose(-2, -1)
        attn_w = attn_w * self.scaler 
        attn_w = attn_w.softmax(dim=-1)
    
        attn_w = attn_w @ v_w
        attn_w = rearrange(attn_w, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn_w = attn_w.mean((2, 3), keepdim=True)

        return attn_h,attn_w

class MSCB(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_sizes=[3,5,7]):
        super(MSCB, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_ch, out_planes=in_ch*6, kernel_size=1)
        self.msdconv = msdconv(in_ch*6,kernel_sizes)
        self.pwconv = BasicConv2d(in_ch*6, out_ch, kernel_size=1,activation=None)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        c1 = self.conv1(x)
        msdc1 = self.msdconv(c1)
        out = self.pwconv(msdc1)
        return x + out
    
class msdconv(nn.Module):
    def __init__(self, in_ch, kernel_sizes):
        super(msdconv, self).__init__()
        self.dconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size,1, kernel_size // 2, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x1,x2,x3=self.dconvs[0](x),self.dconvs[1](x),self.dconvs[2](x)
        return x1+x2+x3

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners) 


#DCERM
class DCERM(nn.Module):
    def __init__(self, x1_ch, x2_ch, x3_ch, x4_ch, out_ch):
        super(DCERM, self).__init__()
        self.threeD_Conv = ThreeD_Conv(x2_ch, x3_ch, x4_ch, x1_ch)
        self.asee = ASEE(in_dim=64,hidden_dim=64)
        self.desc=DESC(in_ch = x1_ch)
        self.conv = BasicConv2d(x1_ch, out_ch, kernel_size=1,activation='silu')

    def forward(self, x1, x2, x3, x4):
        semantic_feature=self.threeD_Conv(x2, x3, x4)
        semantic_feature=F.interpolate(semantic_feature, scale_factor=2, mode='bilinear', align_corners=False)
        edge_feature=self.asee(x1)
        out= self.desc(edge_feature, semantic_feature)
        return out
    
class ThreeD_Conv(nn.Module):
    def __init__(self, x2_ch, x3_ch, x4_ch, out_ch):
        super(ThreeD_Conv, self).__init__()
        self.conv1 = BasicConv2d(x2_ch, out_ch, kernel_size=1,activation='silu')
        self.conv2 = BasicConv2d(x3_ch, out_ch, kernel_size=1,activation='silu')
        self.conv3 = BasicConv2d(x4_ch, out_ch, kernel_size=1,activation='silu')
        self.conv3d = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3),padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.avgpool_3d = nn.AvgPool3d(kernel_size=(3, 1, 1))

    def forward(self, x2, x3, x4):
        x2 = self.conv1(x2)
        x3 = self.conv2(x3)
        x3 = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x4 = self.conv3(x4)
        x4 = F.interpolate(x4, x2.size()[2:], mode='nearest')
        x2_3d = torch.unsqueeze(x2, -3)
        x3_3d = torch.unsqueeze(x3, -3)
        x4_3d = torch.unsqueeze(x4, -3)
        x_fuse = torch.cat([x2_3d, x3_3d, x4_3d], dim=2)
        x_fuse_3d = self.conv3d(x_fuse)
        x_fuse_bn = self.bn(x_fuse_3d)
        x_act = self.leakyrelu(x_fuse_bn)
        x = self.avgpool_3d(x_act)
        x = torch.squeeze(x, 2)
        return x
    
class ASEE(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(in_dim,hidden_dim, 3, padding=1, bias=False),
            norm(hidden_dim),
            act()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.SiLU(inplace=True)
                ))
            self.edge_enhance.append(DPP(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * (width-1), hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        mid = self.in_conv(x)
        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)
            if i == 0:
                out = self.edge_enhance[i](mid)
            else:
                out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)
        out = self.out_conv(out)
        out= x + out
        return out
    
class DPP(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.avgpool = nn.AvgPool2d(3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        edge = x_max - x_avg
        edge = self.out_conv(edge)
        return x+edge
    
class DESC(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.sff=SFF(in_ch = in_ch)
        

    def forward(self, edge_feature, semantic_feature):
        edge_feature, semantic_feature= self.sff(edge_feature,semantic_feature)
        out = self.conv(torch.cat([edge_feature, semantic_feature], dim=1))
        return out

class SFF(nn.Module):
    def __init__(self, in_ch):
        super(SFF,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_ch, in_ch//2, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

        self.conv3 = BasicConv2d(in_ch // 2, in_ch // 2, 1)
        self.conv4 = BasicConv2d(in_ch // 2, in_ch // 2, 1)

        self.w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.silu = nn.SiLU(inplace=True)
        self.conv5 = nn.Conv2d(in_ch//2 , in_ch // 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_ch//2 , in_ch // 2, kernel_size=1, stride=1, padding=0)
      
    def forward(self, x1, semantic_feature):
        edge_feature = self.conv1(x1)
        semantic_feature = self.conv2(semantic_feature)
        
        edge_feature_sig = self.Sigmoid(edge_feature)          #32x88x88 
        semantic_feature_sig = self.Sigmoid(semantic_feature)

        edge_feature = self.conv3(edge_feature)
        semantic_feature = self.conv4(semantic_feature)

        w1 = self.w1
        w2 = self.w2
        
        weight1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        weight2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
           
        edge_feature_1 = self.silu(self.conv5(weight1[0]*edge_feature + weight1[1]*(edge_feature * edge_feature_sig) + weight1[2]*((1 - edge_feature_sig) * semantic_feature_sig * semantic_feature)))
        semantic_feature_1 = self.silu(self.conv6(weight2[0]*semantic_feature + weight2[1]*(semantic_feature * semantic_feature_sig) + weight2[2]*((1 - semantic_feature_sig) * edge_feature_sig * edge_feature)))

        return edge_feature_1, semantic_feature_1

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel//reduction, in_channel, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg = self.conv(self.avg_pool(x))
        max = self.conv(self.max_pool(x))
        att = self.sigmoid(avg + max)
        return x * att
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return self.sigmoid(attn) * x  

class PDecoder(nn.Module):
    def __init__(self, channel):
        super(PDecoder, self).__init__()
        self.relu = nn.ReLU(True)                       
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.ca_x2 = ChannelAttention(channel)
        self.ca_x3 = ChannelAttention(channel)
        self.ca_x4 = ChannelAttention(channel)
        self.ca_x5 = ChannelAttention(channel)
        self.ca_x6 = ChannelAttention(2*channel)
        self.ca_x7 = ChannelAttention(3*channel)
        self.ca_x8 = ChannelAttention(4*channel)
        self.ca_x9 = ChannelAttention(5*channel)
        
        self.sa_x2 = SpatialAttention()
        self.sa_x3 = SpatialAttention()
        self.sa_x4 = SpatialAttention()
        self.sa_x5 = SpatialAttention()
        self.sa_x6 = SpatialAttention()
        self.sa_x7 = SpatialAttention()
        self.sa_x8 = SpatialAttention()
        self.sa_x9 = SpatialAttention()
        
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample9 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample10 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample11 = BasicConv2d(channel, channel, 3, padding=1, activation='silu')
        self.conv_upsample12 = BasicConv2d(2*channel, 2*channel, 3, padding=1, activation='silu')
        self.conv_upsample13 = BasicConv2d(3*channel, 3*channel, 3, padding=1, activation='silu')
        self.conv_upsample14 = BasicConv2d(4*channel, 4*channel, 3, padding=1, activation='silu')
        self.conv_concat1 = BasicConv2d(2*channel, 2*channel, 3, padding=1, activation='silu')
        self.conv_concat2 = BasicConv2d(3*channel, 3*channel, 3, padding=1, activation='silu')
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1, activation='silu')
        self.conv_concat4 = BasicConv2d(5*channel, 5*channel, 3, padding=1, activation='silu')
        self.conv4 = BasicConv2d(5*channel, 5*channel, 3, padding=1, activation='relu')
        self.conv5 = nn.Conv2d(5*channel, 1, 1)

    def forward(self, x1, x2, x3, x4, x5):
        x1_1 = x1  # 32x11x11
        
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x2_1 = self.ca_x2(x2_1)  
        x2_1 = self.sa_x2(x2_1)  

        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * \
               self.conv_upsample3(self.upsample(x2)) * x3
        x3_1 = self.ca_x3(x3_1)  
        x3_1 = self.sa_x3(x3_1)  

        x4_1 = self.conv_upsample4(self.upsample(self.upsample(self.upsample(x1)))) * \
               self.conv_upsample5(self.upsample(self.upsample(x2))) * \
               self.conv_upsample6(self.upsample(x3)) * x4
        x4_1 = self.ca_x4(x4_1)  
        x4_1 = self.sa_x4(x4_1)  

        x5_1 = self.conv_upsample7(self.upsample(self.upsample(self.upsample(x1)))) * \
               self.conv_upsample8(self.upsample(self.upsample(x2))) * \
               self.conv_upsample9(self.upsample(x3)) * \
               self.conv_upsample10(x4) * x5
        x5_1 = self.ca_x5(x5_1)  
        x5_1 = self.sa_x5(x5_1)          

        x2_2 = torch.cat((x2_1, self.conv_upsample11(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat1(x2_2)
        x2_2 = self.ca_x6(x2_2)  
        x2_2 = self.sa_x6(x2_2)  

        x3_2 = torch.cat((x3_1, self.conv_upsample12(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat2(x3_2)
        x3_2 = self.ca_x7(x3_2)  
        x3_2 = self.sa_x7(x3_2)  

        x4_2 = torch.cat((x4_1, self.conv_upsample13(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat3(x4_2)
        x4_2 = self.ca_x8(x4_2)  
        x4_2 = self.sa_x8(x4_2)  

        x5_2 = torch.cat((x5_1, self.conv_upsample14(x4_2)), 1)
        x5_2 = self.conv_concat4(x5_2)
        x5_2 = self.ca_x9(x5_2) 
        x5_2 = self.sa_x9(x5_2)  

        x = self.conv4(x5_2)
        x = self.conv5(x)
        return x

class Upmodel(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(Upmodel, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,      
            stride=1,      
            padding=1,          
            bias=bias          
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        out = self.upsample(x)
        return out


class SDSCNet(nn.Module):
    def __init__(self, channel=32):
        super(SDSCNet, self).__init__()

        self.backbone = pvt_v2_b2()
        path = './model/pvt_v2_b2.pth'       
        save_model = torch.load(path) 
        model_dict = self.backbone.state_dict()         
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}           
        model_dict.update(state_dict)                   
        self.backbone.load_state_dict(model_dict)       

        self.ChannelReduction_1 = BasicConv2d(64, channel, 3, 1, 1, activation=None)  # 64x88x88->32x88x88           
        self.ChannelReduction_2 = BasicConv2d(128, channel, 3, 1, 1, activation=None) # 128x44x44->32x44x44
        self.ChannelReduction_3 = BasicConv2d(320, channel, 3, 1, 1, activation=None) # 320x22x22->32x22x22
        self.ChannelReduction_4 = BasicConv2d(512, channel, 3, 1, 1, activation=None) # 512x11x11->32x11x11
 
        self.pdc1 = PDC(in_ch=32,out_ch=32)
        self.pdc2 = PDC(in_ch=32,out_ch=32)
        self.ecc1 = ECC(in_ch=32, out_ch=32)
        self.ecc2 = ECC(in_ch=32, out_ch=32)

        self.saam1=SAAM(in_ch=32, out_ch=32)
        self.saam2=SAAM(in_ch=32, out_ch=32)
        self.saam3=SAAM(in_ch=32, out_ch=32)   
        self.saam4=SAAM(in_ch=32, out_ch=32)

        self.dcerm = DCERM(x1_ch=64, x2_ch =128, x3_ch=320, x4_ch=512, out_ch=32)

        self.PDecoder = PDecoder(channel)   
        self.upmodel = Upmodel(in_ch=1, out_ch=1)
        self.sigmoid = nn.Sigmoid()        


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        x5=self.dcerm(x1,x2,x3,x4)            #32x88x88

        x1_cr = self.ChannelReduction_1(x1) # 32x88x88
        x2_cr = self.ChannelReduction_2(x2) # 32x44x44
        x3_cr = self.ChannelReduction_3(x3) # 32x22x22
        x4_cr = self.ChannelReduction_4(x4) # 32x11x11

        x1_pdc=self.pdc1(x1_cr)
        x2_pdc=self.pdc2(x2_cr)
        x3_ecc=self.ecc1(x3_cr)
        x4_ecc=self.ecc2(x4_cr)

        x1=self.saam1(x1_pdc)
        x2=self.saam2(x2_pdc)     
        x3=self.saam3(x3_ecc)
        x4=self.saam4(x4_ecc)

        prediction = self.upmodel(self.PDecoder(x4, x3, x2, x1, x5))
        
        return prediction, self.sigmoid(prediction)  