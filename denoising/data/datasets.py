import torch
import glob
import os
import random
from torchvision import transforms
from PIL import Image

class DatasetNoise(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', noise_sigma=50., training=True, crop_size=60, blind_denoising=False, gray_scale=False, max_size=None):
        self.root = root
        self.noise_sigma = noise_sigma
        self.training = training
        self.crop_size = crop_size
        self.blind_denoising = blind_denoising
        self.gray_scale = gray_scale
        self.max_size = max_size

        self._init()

    def _init(self):
        # data paths
        targets = glob.glob(os.path.join(self.root, 'img', '*.*'))[:self.max_size]
        self.paths = {'target' : targets}

        # transforms
        t_list = [transforms.ToTensor()]
        self.image_transform = transforms.Compose(t_list)

    def _get_augment_params(self, size):
        random.seed(random.randint(0, 12345))

        # position
        w_size, h_size = size
        x = random.randint(0, max(0, w_size - self.crop_size))
        y = random.randint(0, max(0, h_size - self.crop_size))

        # flip
        flip = random.random() > 0.5
        return {'crop_pos': (x, y), 'flip': flip}

    def _augment(self, image, aug_params):
        x, y = aug_params['crop_pos']
        image = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        if aug_params['flip']:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, index):
        # target image
        if self.gray_scale:
            target = Image.open(self.paths['target'][index]).convert('L')
        else:
            target = Image.open(self.paths['target'][index]).convert('RGB')

        # transform
        if self.training:
            aug_params = self._get_augment_params(target.size)
            target = self._augment(target, aug_params)
        target = self.image_transform(target)

        # add noise
        if self.blind_denoising:
            noise_sigma = random.randint(0, self.noise_sigma)
        else:
            noise_sigma = self.noise_sigma
        input = target + (noise_sigma / 255.) * torch.randn_like(target)

        return {'input': input, 'target': target, 'path': self.paths['target'][index]}

    def __len__(self):
        return len(self.paths['target'])