import numpy as np
import matplotlib.pyplot as plt
import glob2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import *
from torch.utils.data import DataLoader
from diffusion import *
from net_b import *
from torch.utils.tensorboard import SummaryWriter
import time


train_list = np.load("./train_list_nn_90.npy",allow_pickle=True)
#replace ./ with ../
train_list = [i.replace("./","../") for i in train_list]

test_list = np.load("./test_list_nn_90.npy",allow_pickle=True)
test_list = [i.replace("./","../") for i in test_list]

batch_size=600

dataset_train=DataSet_hyper(train_list)
train_loader=DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
# test_loader=DataLoader(dataset_test,batch_size=batch_size,shuffle=True,num_workers=12,pin_memory=True)

epochs=1000

diff=GaussianDiffusion(Unet(dim=64,dim_mults=(1,1,2,2,4)),image_size=64,p2_loss_weight_k=0,timesteps=500,loss_type="l1").to("cuda:0")

diff=nn.DataParallel(diff,device_ids=[0,1,2])

optimizer=optim.Adam(diff.parameters(),lr=1e-3)#1e-3
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=25,factor=0.75,threshold=0.001, verbose=True,min_lr=1e-8)

name="diff_64_11224_T500_run2_rev"
print("Number of training samples: ",len(train_list))

epoch_done=0

tb=SummaryWriter("./runs/"+name)
# loaded=torch.load("./models/diff_64_11224_T500_seed_31_og_run1.pth")
# # from collections import OrderedDict
# # new_state_dict = OrderedDict()
# # for k, v in loaded["model_state_dict"].items():
# #     name_k = k[7:] # remove `module.`
# #     new_state_dict[name_k] = v


# diff.load_state_dict(loaded["model_state_dict"],strict=False)
# optimizer.load_state_dict(loaded["optimizer_state_dict"])
# scheduler.load_state_dict(loaded["scheduler_state_dict"])
# epoch_done=loaded["epoch"]#331
# loss=loaded["loss"]#0.0021 -- 0.0030
# print("loaded epoch, loss, lr",epoch_done,loss.item())#,optimizer.param_groups[0]["lr"])
 
print("Training:"+name)
# diff.train()

for epoch in (range(epoch_done,epochs)):
    
    st=time.time()
    print("starting epoch: ",epoch)
    l_b=[]
    l_sc=[]

    for inputs,targets,target_n,mx,mn,stack,fs in tqdm(train_loader):
        inputs=inputs.to("cuda:0").float()
        targets=targets.to("cuda:0").float()
        target_n=target_n.to("cuda:0").float()
        
        stack=stack.to("cuda:0")

        optimizer.zero_grad()

        loss=diff(target_n,inputs)

        loss=loss.mean()
        # print(loss.item())

        loss.backward()
        optimizer.step()
        l_b.append(loss.item())

        
    scheduler.step(np.array(l_b).mean())
    print("Epoch: ",epoch,"Loss: ",np.array(l_b).mean(),"Time: ",time.time()-st)
    tb.add_scalar("Loss",np.array(l_b).mean(),epoch)
    tb.add_scalar("LR",optimizer.param_groups[0]["lr"],epoch)


    # torch.save({'epoch': epoch,"model_state_dict":diff.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,"scheduler_state_dict":scheduler.state_dict()}, "./models/"+name+".pth")



