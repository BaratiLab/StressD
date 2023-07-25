import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
# from ema_pytorch import EMA

# from accelerate import Accelerator
from torchinfo import summary

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# CNN encoder
class Encoder(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.cnn1=nn.Conv2d(2, 64, kernel_size=7, stride=2)
        self.cnn2=nn.Conv2d(64, 128, kernel_size=5, stride=1)
        self.cnn3=nn.Conv2d(128, 64, kernel_size=5, stride=1)
        self.pool=nn.AvgPool2d(3, stride=1)
        self.cnn4=nn.Conv2d(64, 16, kernel_size=5, stride=1)

        self.norml=nn.InstanceNorm2d(64)
        self.norm1=nn.InstanceNorm2d(128)
        self.norm2=nn.InstanceNorm2d(64)

        self.atten1=Attention(16)

        self.lin1=nn.Linear(16*15*15, dim*4)

    def forward(self, x):

        # print("x incoder",x.shape)
        x=self.cnn1(x)
        x=F.relu(x)
        x=self.norm1(x)
        
        x=self.cnn2(x)
        x=F.relu(x)
        x=self.norm2(x)

        x=self.cnn3(x)
        x=self.pool(x)
        x=self.norml(x)
        
        x=self.cnn4(x)
        x=self.atten1(x)
        x=self.norml(x)
        # print(x.shape)

        x=x.view(x.size(0),-1)
        # print(x.shape)

        x=self.lin1(x)


        return x


# dim=64
# print(dim*4)

# model=Encoder(64)
# batch_size=12
# summary(model, input_size=(batch_size, 5,32,32), device='cpu')