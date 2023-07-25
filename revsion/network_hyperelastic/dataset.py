import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from sklearn.model_selection import train_test_split
import torch
from torch import nn, einsum
import torch.nn.functional as F
import glob2
import numpy as np
from torch.utils.data import DataLoader


class DataSet_hyper(torch.utils.data.Dataset):
    def __init__(self, file_list):
        
        self.file_list=file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        self.data=np.load(self.file_list[idx])
        
        # print(self.data.shape)

        inputs=torch.from_numpy(self.data[:3])

        inputs[1]=inputs[0]*inputs[1]

        #remove inputs[2]
        inputs=torch.stack([inputs[0],inputs[1]])
        # print(inputs.shape)


        f_s_mx=inputs[1].max()
        f_s_mn=inputs[1].min()

        # f_b_mx=inputs[2].max()
        # f_b_mn=inputs[2].min()
        f_b_mn=torch.tensor(0.0)
        f_b_mx=torch.tensor(0.0)


        targets=torch.from_numpy(self.data[3])
        target_n=targets
        # target_n=target_n

        maxx=target_n.max()
        # minn=target_n.min()
        minn=torch.tensor(0.0)

        if maxx==minn:
            target_n=(target_n-minn)/1e-32
        else:
                
            target_n=(target_n-minn)/(maxx-minn)

        target_n=(target_n*2)-1

        return inputs,targets.unsqueeze(0),target_n.unsqueeze(0),maxx.unsqueeze(0),minn.unsqueeze(0),torch.stack([maxx,minn]),torch.stack([f_s_mx,f_s_mn,f_b_mx,f_b_mn])