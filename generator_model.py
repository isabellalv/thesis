
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

##importing stuff for plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from dcGAN.py import netG
from dcGAN.py import state_dict


## saving model 
##torch.save(netG.state_dict(), 'media/data_cifs/projects/prj_categorization/thesis/gen_output.pth')

## creating model with parameters 
netG = netG(argparse)
netG.load_state_dict(torch.load('media/data_cifs/projects/prj_categorization/thesis/gen_output.pth'))

#Model Inference 
netG.eval()
z = torch.randn(8, 100, 1, 1, device=device)
fake_image = netG(z)
print(fake_image.shape)


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
