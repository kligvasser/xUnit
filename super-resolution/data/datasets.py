import torch
import glob
import os
import random
from torchvision import transforms
from PIL import Image

class DatasetSR(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', scale=4, training=True, crop_size=60, max_size=None):
        self.root = root
        self.scale = scale if (scale % 1) else int(scale)
        self.training = training
        self.crop_size = crop_size
        self.max_size = max_size

        self._init()

    def _init(self):
        # data paths
        inputs = glob.glob(os.path.join(self.root, 'img_x{}'.format(self.scale), '*.*'))[:self.max_size]
        targets = [x.replace('img_x{}'.format(self.scale), 'img') for x in inputs]
        self.paths = {'input' : inputs, 'target' : targets}

        # transforms
        t_list = [transforms.ToTensor()]
        if self.training:
            t_list.append(lambda x: ((255. * x) + torch.zeros_like(x).uniform_(0., 1.)) / 256.,)
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

    def _augment(self, image, aug_params, scale=1):
        x, y = aug_params['crop_pos']
        image = image.crop((x * scale, y * scale, x * scale + self.crop_size * scale, y * scale + self.crop_size * scale))
        if aug_params['flip']:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, index):
        # input image
        input = Image.open(self.paths['input'][index]).convert('RGB')

        # target image
        target = Image.open(self.paths['target'][index]).convert('RGB')

        if self.training:
            aug_params = self._get_augment_params(input.size)
            input = self._augment(input, aug_params)
            target = self._augment(target, aug_params, self.scale)

        input = self.image_transform(input)
        target = self.image_transform(target)

        return {'input': input, 'target': target, 'path': self.paths['target'][index]}

    def __len__(self):
        return len(self.paths['input'])