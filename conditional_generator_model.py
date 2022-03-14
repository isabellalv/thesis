## importing 
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

## CIFAR Generator model
class Generator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                # For input image in CIFAR
                input_deconv = torch.nn.ConvTranspose2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Initializer - with weight biases 
                torch.nn.init.normal(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(input_deconv.bias, 0.0)

                # Batch normalization - for speed and stability 
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation function to learn non-linear relationship 
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # For class label in CIFAR
                label_deconv = torch.nn.ConvTranspose2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)

                # Initializer
                torch.nn.init.normal(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(label_deconv.bias, 0.0)

                # Batch normalization
                self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant(deconv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer of Generator 
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


device = 'cuda:0' #device to run code on 
#ngpu = 1 #number of GPUS to use, don't think I need for cifar 
label_dim = 10 
input_dim = 100
output_dim = 3
num_filters = [512, 256, 128]

## creating model with parameters - change for cifar 
G = Generator(input_dim, label_dim, num_filters, output_dim).to(device)
G.load_state_dict(torch.load('/media/data_cifs/projects/prj_categorization/results/CIFAR_cDCGAN/netG_epoch_45.pth')) #path to generator output specified to epoch that it is loading
    # ^^ specifically loading a epoch within the generator model 
#Model Inference - change for cifar 
G.eval()
#3 rows and 12 columns for both z and c --> is this the right format? 
z = torch.randn(8, 100, 1, 1, device=device) #input to generator, returns a tensor with numbers from norm distri. of size 8 (8 images)
c = torch.randn(8, 10, 1, 1, device=device) #both the class and the image seem to have the same dimensions
fake_image = G(z, c) #need to pass in z and c, need to know the dimensions of those inputs, need to be of a specific type 
    #do I need to pass in c, since c has the same number of elements? 
print(fake_image.shape) #print output value shape 


#in training code, need to figure out what is it that is being inputed to the generator (put ipdb where the generator is called) 
# do something .shape to get shape of Z
#feeding it random noise and a class label, and it will give you an image

fake_image1 = torch.transpose(torch.transpose(fake_image[2].squeeze(),1,0),2,1).detach().cpu().numpy() #this is reshaping so color channel is last 
#fake_image1 = fake_image[2][0].squeeze().detach().cpu().numpy()
print(fake_image1.shape) #brackets in fake_images indicates what image in the sequence of 8 you want to show and squeeze removes input of size 1 removed
#plt.show()
#import ipdb; ipdb.set_trace()

mn, mx = fake_image1.min(), fake_image1.max()
fake_image1 = (fake_image1 - mn)/(mx - mn)

plt.imshow((fake_image1 * 255.).astype(np.uint8)) 
plt.savefig('test1.png') #assigning name to figure 

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