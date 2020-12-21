# xUnit
Learning a Spatial Activation Function for Efficient Image Restoration.

<p align="center">
  <img width="1004" height="414" src="/figures/activations.png">
</p>

Please refer our [paper](https://arxiv.org/abs/1711.06445) for more details.


## Citation
If you use this code for your research, please cite our papers:

```
@inproceedings{kligvasser2018xunit,
  title={xunit: Learning a spatial activation function for efficient image restoration},
  author={Kligvasser, Idan and Rott Shaham, Tamar and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2433--2442},
  year={2018}
}
```

```
@article{kligvasser2018dense,
  title={Dense xUnit Networks},
  author={Kligvasser, Idan and Michaeli, Tomer},
  journal={arXiv preprint arXiv:1811.11051},
  year={2018}
}
```

## Code

### Clone repository

Clone this repository into any place you want.

```
git clone https://github.com/kligvasser/xUnit
cd xUnit
```

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code requires PyTorch 1.0+ and python 3+.

### Super-resoltution
Pretrained models are avaible at: [LINK](https://www.dropbox.com/s/l6fmgn1r600dmyq/sr_pretrained.zip?dl=0).

#### Dataset preparation
For the super-resolution task, the dataset should contains a low and high resolution pairs, in folder structure of:

```txt
train
├── img
├── img_x2
├── img_x4
val
├── img
├── img_x2
├── img_x4
```

You may prepare your own data by using the matlab script:

```
./super-resolution/scripts/matlab/bicubic_subsample.m
```

Or download a prepared dataset based on the [BSD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) datasets from [LINK](https://www.dropbox.com/s/o1nzpr9q7vup8b7/bsdvoc.zip?dl=0).

#### Train xSRGAN x4 PSNR model
```
python3 main.py --root <path-to-dataset> --g-model g_xsrgan --d-model d_xsrgan --model-config "{'scale':4, 'gen_blocks':10, 'dis_blocks':5}" --scale 4 --reconstruction-weight 1.0 --perceptual-weight 0 --adversarial-weight 0 --crop-size 40
```

#### Train xSRGAN x4 WGAN-GP model
```
python3 main.py --root <path-to-dataset> --g-model g_xsrgan --d-model d_xsrgan --model-config "{'scale':4, 'gen_blocks':10, 'dis_blocks':5}" --scale 4 --reconstruction-weight 1.0 --perceptual-weight 1.0 --adversarial-weight 0.01 --crop-size 40 --epochs 2000 --step-size 800 --gen-to-load <path-to-psnr-pretrained-pt> --wgan --penalty-weight 10
```

#### Train xSRGAN x4 with SN-discriminator model
```
python3 main.py --root <path-to-dataset> --g-model g_xsrgan --d-model d_xsrgan --model-config "{'scale':4, 'gen_blocks':10, 'dis_blocks':5, 'spectral':True}" --scale 4 --reconstruction-weight 1.0 --perceptual-weight 1.0 --adversarial-weight 0.01 --crop-size 40 --epochs 2000 --step-size 800 --gen-to-load <path-to-psnr-pretrained-pt> --dis-betas 0 0.9
```

#### Eval xSRGAN x4 model
```
python3 main.py --root <path-to-dataset> --g-model g_xsrgan --d-model d_xsrgan --model-config "{'scale':4, 'gen_blocks':10, 'dis_blocks':5}" --scale 4 --evaluation --gen-to-load <path-to-pretrained-pt>
```

### Gaussian denoising
Pretrained models are avaible at: [LINK]().

#### Dataset preparation
For the denoising task, the dataset should contains only clean images, in folder structure of:

```txt
train
├── img
val
├── img
```

#### Train xDNCNN Grayscale 50 sigma PSNR model
```
python3 main.py --root <path-to-dataset> --g-model g_xdncnn --d-model d_xdncnn --model-config "{'gen_blocks':10, 'dis_blocks':4, 'in_channels':1}" --reconstruction-weight 1.0 --perceptual-weight 0 --adversarial-weight 0 --crop-size 50 --gray-scale --noise-sigma 50 --epochs 500 --step-size 150
```

#### Train xDNCNN Grayscale 50 sigma WGAN-GP model
```
python3 main.py --root <path-to-dataset> --g-model g_xdncnn --d-model d_xdncnn --model-config "{'gen_blocks':10, 'dis_blocks':4, 'in_channels':1}" --reconstruction-weight 1.0 --perceptual-weight 1.0 --adversarial-weight 0.01 --crop-size 64 --gray-scale --noise-sigma 50 --epochs 1000 --step-size 300 --gen-to-load <path-to-psnr-pretrained-pt> --wgan --penalty-weight 10
```

#### Train xDNCNN Grayscale blind PSNR model
```
python3 main.py --root <path-to-dataset> --g-model g_xdncnn --d-model d_xdncnn --model-config "{'gen_blocks':10, 'dis_blocks':5, 'in_channels':1}" --reconstruction-weight 1.0 --perceptual-weight 0 --adversarial-weight 0 --crop-size 50 --gray-scale --noise-sigma 50 --blind --epochs 500 --step-size 150

```