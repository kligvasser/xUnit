
### imports ###
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
import torchvision.transforms as transforms

### defines ###

### classes ###
class data_set(Dataset):
    def __init__(self,data_dir="",scale=4,pixels_shuffle=False,transform=None):
        self.data_dir = data_dir
        self.scale = scale
        self.transform = transform
        if (pixels_shuffle):
            self.imgs_list = glob.glob(data_dir+('gt_ps_im/*x%s.*'%scale))
        else:
            self.imgs_list = glob.glob(data_dir+('gt_im/*x%s.*'%scale))
        self.num_imgs = len(self.imgs_list)
    def __getitem__(self,idx):
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        name_in,name_tar = self.get_file_name(idx)
        img_in = Image.open(name_in)
        img_tar = Image.open(name_tar)
        if self.transform is not None:
            random.seed(seed)
            img_in = self.transform(img_in)
            random.seed(seed)
            img_tar = self.transform(img_tar)
        return img_in,img_tar
    def __len__(self):
        return len(self.imgs_list)
    def get_file_name(self,idx):
        name_tar = self.imgs_list[idx]
        name_in = name_tar.replace('gt','lr')
        return name_in,name_tar

### functions ###
def plot_images(images,use_cuda=False):
    imgs2plot = tensor_to_cpu(images,use_cuda)
    fig,axes = plt.subplots(1,3)
    fig.subplots_adjust(hspace=0.6,wspace=0.1)
    labels = ['Original','LR','Reconstructed']
    for i,ax in enumerate(axes.flat):
        img = tensor_to_img(imgs2plot[i,:,:,:])
        ax.imshow(img,cmap='gray')
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

def tensor_to_img(tensor):
    unloader = transforms.Compose([transforms.ToPILImage()])
    img = unloader(tensor)
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

def prepare_data(input,target,pixels_shuffle=False,use_cuda=True,volatile=True):
    if (use_cuda):
        input = Variable(input.cuda(),volatile=volatile)
        target = Variable(target.cuda())
    else:
        input = Variable(input,volatile=volatile)
        target = Variable(target)
    if (pixels_shuffle):
        return input,target
    else:
        input = rgb_to_ycbcr(input)
        target = rgb_to_ycbcr(target)
        return input,target

def resize_lr(image,size,use_cuda):
    loader = transforms.Compose([transforms.ToPILImage(),transforms.Resize(size,interpolation=Image.NEAREST),transforms.ToTensor()])
    image = image.squeeze(0)
    image = loader(tensor_to_cpu(image.data,use_cuda))
    image = image.unsqueeze(0)
    image = torch.autograd.Variable(tensor_to_gpu(image,use_cuda))
    return image

def rgb_to_ycbcr(input):
    output = Variable(input.data.new(*input.size()))
    if (input.size()[1]==3):
        output[:,0,:,:] = input[:,0,:,:]*(65.481/255.0)+input[:,1,:,:]*(128.553/255.0)+input[:,2,:,:]*(24.966/255.0)+(16/255.0)
    else:
        output[:,0,:,:] = input[:,0,:,:]*(65.481/255.0)+input[:,0,:,:]*(128.553/255.0)+input[:,0,:,:]*(24.966/255.0)+(16/255.0)
    output = output[:,0:1,:,:]
    return output

### main ###


