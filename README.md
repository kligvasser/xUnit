# xUnit
Learning a Spatial Activation Function for Efficient Image Restoration.

![xUnit]({{site.baseurl}}/figures/xunit_relu_scheme.png)


Please refer our [paper](https://arxiv.org/abs/1711.06445) for more details.


# Dependencies
- python (tested with 3.5)
- PyTorch >= 0.2.0
- [PyINN](https://github.com/szagoruyko/pyinn)


# Code
Clone this repository into any place you want.
	
	git clone https://github.com/kligvasser/xUnit
	cd xUnit


# Results
### Gaussian Denoising

The average PSNR in [dB] attained by several state of the art denoising algorithms on the BSD68:

| Methods | BM3D | WNNM | EPLL | MLP | DnCNN-S | xDnCNN |
|   ---   | ---  | ---  | ---  | --- |   ---   |  ---   |
| # Parameters | - | - | - | - | 555K | 303K |
| sigma=25 | 28.56 | 28.82 | 28.68 | 28.95 | 29.22 | 29.21 |
| sigma=50 | 25.62 | 25.87 | 25.67 | 26.01 | 26.23 | 26.26 |

