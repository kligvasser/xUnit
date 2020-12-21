import os
from .datasets import  DatasetSR
from torch.utils.data import DataLoader

def get_loaders(args):
    # datasets
    dataset_train = DatasetSR(root=os.path.join(args.root, 'train'), scale=args.scale, training=True, crop_size=args.crop_size)
    dataset_val = DatasetSR(root=os.path.join(args.root, 'val'), scale=args.scale, training=False, max_size=args.max_size)

    # loaders
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader_eval = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
    loaders = {'train': loader_train, 'eval': loader_eval}

    return loaders
