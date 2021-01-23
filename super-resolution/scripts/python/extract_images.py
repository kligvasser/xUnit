import argparse
import glob
import os
import numpy as np
from PIL import Image
from multiprocessing import Pool

def get_arguments():
    parser = argparse.ArgumentParser(description='Extract sub images for faster data loading.')
    parser.add_argument('--root', default='', required=True, help='source folder')
    parser.add_argument('--crop-size', default=256, type=int, help='crop size (default: 256)')
    parser.add_argument('--step-size', default=128, type=int, help='step size (default: 128)')
    parser.add_argument('--scale', default=4, type=int, help='super-resolution scale (default: 4)')
    parser.add_argument('--num-threads', default=8, type=int, help='num of threads (default: 8)')
    parser.add_argument('--only-once', default=False, action='store_true', help='center sub image')
    args = parser.parse_args()
    return args

def mkdir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def mkdirs(args):
    mkdir(os.path.join(args.root,'img_sub'))
    mkdir(os.path.join(args.root,'img_sub_x{}'.format(args.scale)))

def get_image_paths(args):
    paths = glob.glob(os.path.join(args.root, 'img_x{}'.format(args.scale), '*.*'))
    return paths

def save_images(args, i, path, input, target):
    save_path = path.replace('img_x', 'img_sub_x').replace('.png', '_{}.png'.format(i))
    input.save(save_path)
    save_path = path.replace('img_x{}'.format(args.scale), 'img_sub').replace('.png', '_{}.png'.format(i))
    target.save(save_path)

def load_images(path, args):
    input = Image.open(path)
    target = Image.open(path.replace('img_x{}'.format(args.scale), 'img'))
    return input, target

def worker(path, args):
    input, target = load_images(path, args)

    h, w = input.size
    hs = np.arange(0, h - args.crop_size + 1, args.step_size)
    ws = np.arange(0, w - args.crop_size + 1, args.step_size)

    hs = np.append(hs, h - (args.crop_size + 1))
    ws = np.append(ws, w - (args.crop_size + 1))

    counts = 0

    if args.only_once:
        hs = [hs[len(hs) // 2]]
        ws = [ws[len(ws) // 2]]

    for x in hs:
        for y in ws:
            cropped_input = input.crop((x, y, (x + args.crop_size), (y + args.crop_size)))
            cropped_target = target.crop((x * args.scale, y * args.scale, (x + args.crop_size) * args.scale, (y + args.crop_size) * args.scale))
            save_images(args, counts, path, cropped_input, cropped_target)
            counts += 1

    print('Processed {:s}'.format(os.path.basename(path)))

def main():
    args = get_arguments()

    mkdirs(args)
    paths = get_image_paths(args)
    pool = Pool(args.num_threads)

    for path in paths:
        pool.apply_async(worker, args=(path, args))

    pool.close()
    pool.join()
    print('All subprocesses done.')

if __name__ == "__main__":
    main()
