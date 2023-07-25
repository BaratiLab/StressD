import glob2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from diffusion import *
from net_b import *
from dataset import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import sys
from vit_min_max import *
import torch.nn.init as init
from collections import OrderedDict
import argparse
import os
import time

parser=argparse.ArgumentParser()

parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--sampling_timesteps",type=int,default=50)
parser.add_argument("--image_size",type=int,default=64)
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--location",type=str,default="run1")

args = parser.parse_args()

device=args.device
batch_size=args.batch_size
image_size=args.image_size


if not os.path.exists("./preds_res_outa/"+args.location):
    os.makedirs("./preds_res_outa/"+args.location)

    # /home/yayati/stressD/gdrive/Research_PHD/StressD/results

location="./preds_res_outa/"+args.location+"/"
sampling_timesteps=args.sampling_timesteps


print("Device: ",device,"\nBatch Size: ",batch_size,"\nSampling Timesteps: ",args.sampling_timesteps,"\nImage Size: ",args.image_size,"\nLocation: ",args.location)


# data=np.load('../data/all_data_m-002.npy')
# train,test=train_test_split(data,test_size=0.2,random_state=42)


# dataset_test=DataSet_og(test)
# print("Number of batches: ",len(dataset_test)//batch_size)
# test_loader=DataLoader(dataset_test,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)


# test_list = np.load("./test_list_nn_90.npy",allow_pickle=True)
# test_list = [i.replace("./","../") for i in test_list]

test_list = glob2.glob("../results_8/**.npy")

dataset_test=DataSet_hyper(test_list)

test_loader=DataLoader(dataset_test,batch_size=batch_size,shuffle=True,pin_memory=True)

diff=GaussianDiffusion(Unet(dim=64,dim_mults=(1,1,2,2,4)),image_size=24,p2_loss_weight_k=0,timesteps=500,loss_type="l1",train=False,sampling_timesteps=sampling_timesteps).to(device)

min_max=VisionTransformer(embed_dim=256,hidden_dim=512,num_heads=8,num_layers=16,num_channels=2,patch_size=4,num_patches=256).to("cpu")

diff_states=torch.load("./models/models_new/diff_64_11224_T500_run2_rev.pth",map_location=device)
print("Diffusion Trained for: ",diff_states["epoch"])

min_max_states=torch.load("./models/models_new/vit_min_max_clip_L1_run_new_new_data_3.pth",map_location="cpu")
print("Min Max Trained for: ",min_max_states["epoch"])

new_state_dict = OrderedDict()
new_state_dict_vit = OrderedDict()

for k, v in diff_states["model_state_dict"].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

for k, v in min_max_states["model_state_dict"].items():
    name = k[7:] # remove `module.`
    new_state_dict_vit[name] = v

diff.load_state_dict(new_state_dict)
min_max.load_state_dict(new_state_dict_vit)

def test_looper(model,test_loader,device,save_path,image_size):
    
    with torch.no_grad():
        for i, (inps,target,target_n,maxx,minn,stack,f_s) in tqdm(enumerate(test_loader)):

            print("\nDoing batch {}\n".format(i))
            st=time.time()
            inps=inps.to(device).float()
            target=target.to(device).float()
            b=target.shape[0]

            f_s=f_s.to("cpu").float()

            pred=model((b,1,image_size,image_size),inps)
            pred_min_max=min_max(inps.to("cpu"),f_s)
            print("Time taken: ",time.time()-st)

            #save to dict
            save_dict={}

            save_dict["inps"]=inps.detach().cpu().numpy()
            save_dict["target"]=target.detach().cpu().numpy()
            save_dict["target_n"]=target_n.detach().cpu().numpy()
            save_dict["stack"]=stack.detach().cpu().numpy()
            save_dict["pred"]=pred.detach().cpu().numpy()
            save_dict["pred_min_max"]=pred_min_max.detach().cpu().numpy()

            # np.save(save_path+"batch_{}".format(i),save_dict)


test_looper(diff,test_loader,device,location,image_size)
