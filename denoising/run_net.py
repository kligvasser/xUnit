
### imports ###
import torch
import functions
import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse

### defines ###
use_cuda = torch.cuda.is_available()

### functions ###
def run_net(plot_images=False):
    psnrs = []
    cnn.eval()
    for i,(images,labels) in enumerate(tst_loader):
        images,labels = functions.prepare_data(images,labels,use_cuda=use_cuda)
        outputs= cnn(images)
        cleans = images-outputs # res
        cleans = cleans.clamp(min=0,max=1)
        psnr = compute_psnr(cleans,labels)
        psnrs.append(psnr)
        if (plot_images):
            functions.plot_images(torch.cat((labels.data,images.data,cleans.data),0),use_cuda,num_cols=3)
    return psnrs

def compute_psnr(x,y):
    x = (functions.tensor_to_cpu(x.data[0,0,:,:],use_cuda)).numpy()
    y = (functions.tensor_to_cpu(y.data[0,0,:,:],use_cuda)).numpy()
    err = np.sum((x.astype("float")-y.astype("float"))**2)
    err /= float(x.shape[0]*x.shape[1])
    psnr = -10*np.log10(err)
    return psnr

def init_net(opt):
    net2load = './checkpoints/xdncnn_res_f64k3k9_s'+str(opt.sigma)+'.pkl'
    cnn = models.xDnCNN(64)
    cnn.load_state_dict(torch.load(net2load))
    cnn = functions.tensor_to_gpu(cnn,use_cuda)
    return cnn

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',help='path to dataset',default='./data/bsd68/')
    parser.add_argument('--plot',help='plot images',default=False,action='store_true')
    parser.add_argument('--sigma',type=int,help='noise sigma 25/50',default=25)
    parser.add_argument('--manual_seed',type=int,help='manual seed')
    opt = parser.parse_args()
    if (opt.manual_seed is None):
        opt.manual_seed = np.random.randint(1,10000)
    return opt

### main ###
opt = get_arguments()
torch.manual_seed(opt.manual_seed)
cnn = init_net(opt)
tst_dset = functions.data_set(imgs_lib=opt.images,noise_std=(opt.sigma/255.0),transform=transforms.Compose([transforms.ToTensor()]))
tst_loader = DataLoader(tst_dset,batch_size=1,shuffle=False,num_workers=1)
psnrs = run_net(plot_images=opt.plot)
print('Sigma: %d ,Average PSNR %2.3f'%(opt.sigma,(sum(psnrs)/float(len(psnrs)))))