
### imports ###
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import matplotlib.image as mpimg
import smtplib
from PIL import Image
from torch.utils.data.dataset import Dataset
from random import randint
import os

### defines ###

### classes ###
class data_set(Dataset):
    def __init__(self,imgs_lib,noise_std,transform=None):
        self.img_path = imgs_lib
        self.noise_std = noise_std
        self.transform = transform
        self.imgs_name = glob.glob(imgs_lib+"*.*")
        self.n_imgs = len(self.imgs_name)

    def __getitem__(self,index):
        img2read = self.imgs_name[index]
        img = Image.open(img2read)
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        label = img
        noise_std = self.noise_std
        img = img+noise_std*torch.randn(img.size()).type(torch.FloatTensor)
        return img,label

    def __len__(self):
        return len(self.imgs_name)

### functions ###
def plot_images(images,use_cuda=False,num_cols=3):
    imgs2plot = tensor_to_cpu(images,use_cuda)
    num_imgs = imgs2plot.size()[0]-1
    num_rows = 1+num_imgs//num_cols
    fig,axes = plt.subplots(num_rows,num_cols)
    fig.subplots_adjust(hspace=0.6,wspace=0.1)
    labels = ['Original','Noisy','xDnCNN']
    for i,ax in enumerate(axes.flat):
        img = tensor_to_img(imgs2plot[i,:,:,:])
        ax.imshow(img[:,:,0],cmap='gray')
        label2plot = (labels[i])
        ax.set_xlabel(label2plot)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def count_model_weights(model):
    tot = 0
    for m in model.modules():
        if isinstance(m,torch.nn.Conv2d):
            tot += (m.weight.data.size()[0]*m.weight.data.size()[1]*m.weight.data.size()[2]*m.weight.data.size()[3])
        elif isinstance(m,torch.nn.BatchNorm2d):
            tot += m.weight.data.size()[0]+m.bias.data.size()[0]
        else:
            continue
    return tot

def prepare_data(input,target,use_cuda=True,volatile=True):
    if (use_cuda):
        input = Variable(input.cuda(),volatile=volatile)
        target = Variable(target.cuda())
    else:
        input = Variable(input,volatile=volatile)
        target = Variable(target)
    return input,target

def tensor_to_img(t):
    if (len(t.shape)==4):
        t = t[0,:,:,:]
    img = t.numpy().transpose((1,2,0))
    return img

def tensor_to_cpu(tensor,is_cuda):
    if (is_cuda):
        return tensor.cpu()
    else:
        return tensor

def tensor_to_gpu(tensor,is_cuda):
    if (is_cuda):
        return tensor.cuda()
    else:
        return tensor

### main ###
