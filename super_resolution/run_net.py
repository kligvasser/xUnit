
### imports ###
import torch
import functions
import models
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

### defines ###
use_cuda = torch.cuda.is_available()

### functions ###
def run_net(opt):
    psnrs = []
    cnn.eval()
    for i,(images,labels) in enumerate(tst_dloader):
        images,labels = functions.prepare_data(images,labels,pixels_shuffle=opt.pixels_shuffle,use_cuda=use_cuda)
        outputs = cnn(images)
        outputs.data = torch.clamp(outputs.data,min=0,max=1)
        psnr = compute_psnr(outputs,labels,opt)
        psnrs.append(psnr)
        if (opt.plot):
            if (opt.pixels_shuffle):
                images = functions.resize_lr(images,(outputs.size()[2],outputs.size()[3]),use_cuda)
                functions.plot_images(torch.cat((labels.data,images.data,outputs.data),0),use_cuda=use_cuda)
            else:
                functions.plot_images(torch.cat((labels.data,images.data,outputs.data),0),use_cuda=use_cuda)
    return psnrs

def init_net(opt):
    if (opt.model=='xsrcnnf'):
        net2load = './checkpoints/xsrcnnf_s'+str(opt.scale)+'_k9k3k5.pkl'
        cnn = models.xSRCNNf()
    elif (opt.model=='xsrcnnc'):
        net2load = './checkpoints/xsrcnnc_s'+str(opt.scale)+'_c42c32.pkl'
        cnn = models.xSRCNNc()
    else:
        net2load = './checkpoints/xsrresnet_s4_10b.pkl'
        cnn = models.xSRResNet()
    cnn.load_state_dict(torch.load(net2load))
    cnn = functions.tensor_to_gpu(cnn,use_cuda)
    return cnn

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',help='path to dataset',default='./data/bsd100/')
    parser.add_argument('--plot',help='plot images',default=False,action='store_true')
    parser.add_argument('--scale',type=int,help='super-resolution scale 3/4',default=4)
    parser.add_argument('--model',type=str,help='super-resoltuion model xsrcnnf/xsrcnnc/xsrresnet',default='xsrcnnf')
    opt = parser.parse_args()
    if (opt.model=='xsrresnet'):
        opt.scale = 4
        opt.pixels_shuffle = True
    else:
        opt.pixels_shuffle = False
    return opt

def compute_psnr(x,y,opt):
    shave_size = opt.scale
    if (opt.pixels_shuffle):
        x = functions.rgb_to_ycbcr(x)
        y = functions.rgb_to_ycbcr(y)
    x = (functions.tensor_to_cpu(x.data[0,0,shave_size:-shave_size,shave_size:-shave_size],use_cuda)).numpy()
    y = (functions.tensor_to_cpu(y.data[0,0,shave_size:-shave_size,shave_size:-shave_size],use_cuda)).numpy()
    err = np.sum((x.astype("float")-y.astype("float"))**2)
    err /= float(x.shape[0]*x.shape[1])
    psnr = -10*np.log10(err)
    return psnr

### main ###
opt = get_arguments()
cnn = init_net(opt)
tst_dset = functions.data_set(data_dir=opt.images,scale=opt.scale,pixels_shuffle=opt.pixels_shuffle,transform=transforms.Compose([transforms.ToTensor()]))
tst_dloader = DataLoader(tst_dset,batch_size=1,shuffle=False,num_workers=1)
psnrs = run_net(opt)
print("Scale: x%s, Average PSNR %2.3f"%(opt.scale,(sum(psnrs)/float(len(psnrs)))))