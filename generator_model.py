
## importing pytorch - is this necessary? 
from __future__ import print_function
import os, random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

##importing stuff for plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML

class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=3): #nc=3 for color image, nc=1 for grayscale
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

## saving model 
##torch.save(netG.state_dict(), 'media/data_cifs/projects/prj_categorization/thesis/gen_output.pth')
device = 'cuda:0' #device to run code on 
ngpu = 1 #number of GPUS to use 

## creating model with parameters 
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load('/media/data_cifs/projects/prj_categorization/cifar_gen_output/netG_epoch_25.pth')) #path to generator output specified to epoch that it is loading

#Model Inference 
netG.eval()
z = torch.randn(8, 100, 1, 1, device=device) #input to generator, returns a tensor with numbers from norm distri. of size 8 (8 images)
fake_image = netG(z) 
print(fake_image.shape) #print output value shape 

fake_image1 = torch.transpose(torch.transpose(fake_image[5].squeeze(),1,0),2,1).detach().cpu().numpy() #this is reshaping so color channel is last 
print(fake_image1.shape) #brackets in fake_images indicates what image in the sequence of 8 you want to show and squeeze removes input of size 1 removed
#plt.show()
import ipdb; ipdb.set_trace()

plt.imshow(fake_image1) 
plt.savefig('test.png') #assigning name to figure 

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
