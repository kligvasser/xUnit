# xUnit
Learning a Spatial Activation Function for Efficient Image Restoration.

<p align="center">
  <img width="495" height="396" src="/figures/figure1.png">
</p>

Please refer our [paper](https://arxiv.org/abs/1711.06445) for more details.


# Dependencies
- python (tested with 3.5)
- PyTorch >= 0.2.0


# Code
Clone this repository into any place you want.

	git clone https://github.com/kligvasser/xUnit
	cd xUnit


# Results
### Gaussian Denoising

The average PSNR in [dB] attained by several state of the art denoising algorithms on the BSD68:

| Methods | BM3D | WNNM | EPLL | MLP | DnCNN-S | xDnCNN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| # Parameters | - | - | - | - | 555K | **303K** |
|      σ=25    | 28.56 | 28.82 | 28.68 | 28.95 | **29.22** | 29.21 |
|      σ=50    | 25.62 | 25.87 | 25.67 | 26.01 | 26.23 | **26.26** |


### Single Image Super Resolution

The average PSNR in [dB] attained in the task of 3× and 4× SR on BSD100 dataset:

| Methods | SRCNN | xSRCNN-c | xSRCNN-f |
| :---: | :---: | :---: | :---: |
| # Parameters | 57K | 44K | **32K** |
| 3× | 28.41 | **28.54** | 28.53 |
| 4× | 26.90 | 27.04 | **27.06** |

The average PSNR in [dB] attained in the task of 4× SR on BSD100 dataset:

| Methods | SRResNet | xSRResNet |
| :---: | :---: | :---: |
| # Parameters | 1.546M | **1.155M** |
| 4× | 27.58 | **27.61** |
