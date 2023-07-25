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


train_list = np.load("./train_list_nn_90.npy",allow_pickle=True)
#replace ./ with ../
train_list = [i.replace("./","../") for i in train_list]

test_list = np.load("./test_list_nn_90.npy",allow_pickle=True)
test_list = [i.replace("./","../") for i in test_list]

dataset_train=DataSet_hyper(train_list)
dataset_test=DataSet_hyper(test_list)

print(len(dataset_train),len(dataset_test))
    
batch_size=750
epochs=1001

train_loader=DataLoader(dataset_train,batch_size=batch_size,shuffle=True,pin_memory=True)
test_loader=DataLoader(dataset_test,batch_size=batch_size,shuffle=True,pin_memory=True)

name="vit_min_max_clip_L1_run_new_new_data_2"
tb=SummaryWriter("./runs/"+name)
print("Training:"+name)

# net=VisionTransformer(embed_dim=256,hidden_dim=512,num_heads=8,num_layers=6,num_channels=5,patch_size=2,num_patches=256).to("cuda:0") 

net=VisionTransformer(embed_dim=256,hidden_dim=512,num_heads=8,num_layers=16,num_channels=2,patch_size=4,num_patches=256).to("cuda:0") 

for m in net.modules():
    if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


optimizer=optim.Adam(net.parameters(),lr=1e-3)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=25,factor=0.75,threshold=0.1, verbose=True,min_lr=1e-8)

epoch_done=0


# loaded_model=torch.load("./models/vit_min_max_clip_L1_run_new_new_data_2.pth")
# # net.load_state_dict(loaded_model["model_state_dict"])

# new_state_dict = OrderedDict()

# for k, v in loaded_model["model_state_dict"].items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v

# net.load_state_dict(new_state_dict)

# optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
# scheduler.load_state_dict(loaded_model["scheduler_state_dict"])
# epoch_done=loaded_model["epoch"]
# loss=loaded_model["loss"]
# print("Loaded model from epoch: ",epoch_done,"Loss: ",loss)

net=nn.DataParallel(net,device_ids=[0,1,2])

l1loss=nn.L1Loss(reduction="mean")
mse=nn.MSELoss(reduction="mean")

criterion=l1loss

for epoch in range(epoch_done,epochs):
    st=time.time()
    l_b=[]
    l_mse=[]
    ll=[]
    net.train()

    for inputs,targets,targets_n,maxx,minn, stack,f_s in tqdm(train_loader):
        
        inputs=inputs.to("cuda:0").float()
        stack=stack.to("cuda:0").float()

        f_s=f_s.to("cuda:0").float()

        outputs=net(inputs,f_s)

        optimizer.zero_grad()

        loss=criterion(outputs,stack)
        # print(loss)

        ms=mse(outputs,stack)
        l1=l1loss(outputs,stack)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 100)

        optimizer.step()
        ll.append(loss.item())
        l_b.append(l1.item())
        l_mse.append(ms.item())

    scheduler.step(np.array(l_b).mean())


    torch.save({'epoch': epoch,"model_state_dict":net.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,"scheduler_state_dict":scheduler.state_dict()}, "./models/"+name+".pth")
    print("\nEpoch:",epoch,"Loss:",np.array(l_b).mean(),"MSE LOSS: ",np.array(l_mse).mean(),"Time:",time.time()-st,"min_max: ",outputs[0].detach().cpu().numpy(),"GT: ",stack[0].detach().cpu().numpy(),"\n")

    #validation loop every 10 eopch
    if epoch%50==0:
        with torch.no_grad():
            net.eval()
            
            print("\n Validating ...\n")
            l_b_t=[]
            l_mse_t=[]
            ll_t=[]
            for inputs,targets,targets_n,maxx,minn,stack,f_s in tqdm(test_loader):
                
                inputs=inputs.to("cuda:0").float()
                stack=stack.to("cuda:0").float()

                f_s=f_s.to("cuda:0").float()

                outputs=net(inputs,f_s)

                # ms=mse(outputs,stack)

                mae_loss = torch.mean(torch.abs(outputs - stack)).detach().cpu().numpy()
                mse_loss= torch.mean((outputs - stack)**2).detach().cpu().numpy()

                l_b_t.append(mae_loss)
                l_mse_t.append(mse_loss)
            print("Validation Loss:",np.array(l_b_t).mean(),"MSE LOSS: ",np.array(l_mse_t).mean())
            tb.add_scalar("Validataion Loss",np.array(l_b_t).mean(),epoch)

    
    tb.add_scalar("Loss",np.mean(np.array(ll)),epoch)
    tb.add_scalar("Learning Rate",optimizer.param_groups[0]['lr'],epoch)
