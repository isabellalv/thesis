
## importing pytorch - is this necessary? 
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


##import torch 
##import from dcGAN.py

## saving model 
torch.save(netG.state_dict(), ../prj_categorizaation)

## creating model with parameters 
netG = netG (*args *kwargs)
netG.load_state_dict(torch.load(../prj_categorizaation))
netG.eval()


# POSSIBLE WAY TO PLOT: **Visualization of G’s progression**
# previous model saved the generator’s output on the fixed_noise batch
# after every epoch of training. Now, we can visualize the training
# progression of G with an animation. Press the play button to start the
# animation.
# 

#%%capture
# fig = plt.figure(figsize=(8,8))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# HTML(ani.to_jshtml())
