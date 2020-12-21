import argparse
import glob
import os
from PIL import Image, ImageStat
from multiprocessing import Pool

def get_arguments():
    parser = argparse.ArgumentParser(description='Remove images in dataset')
    parser.add_argument('--root', default='', required=True, help='source folder')
    parser.add_argument('--scale', default=4, type=int, help='super-resolution scale (default: 4)')
    parser.add_argument('--limit-size', default=50, type=int, help='limit size for input (default: 50)')
    parser.add_argument('--num-threads', default=8, type=int, help='num of threads (default: 8)')
    args = parser.parse_args()
    return args

def is_grayscale(image):
    stat = ImageStat.Stat(image)

    if sum(stat.sum) / 3 == stat.sum[0]:
        return True
    else:
        return False

def is_small(image, args):
    h, w = image.size

    if h < args.limit_size or w < args.limit_size:
        return True
    else:
        return False

def remove(path):
    if os.path.isfile(path):
        os.remove(path)

def get_image_paths(args):
    paths = glob.glob(os.path.join(args.root, 'img_x{}'.format(args.scale), '*.*'))
    return paths

def get_paths(path, args):
    input = path
    target = path.replace('img_x{}'.format(args.scale), 'img')
    return input, target

def worker(path, args):
    input, target = get_paths(path, args)

    image = Image.open(input).convert("RGB")

    if is_grayscale(image) or is_small(image, args):
        remove(input)
        remove(target)

        print('Removed {:s}'.format(os.path.basename(path)))

def main():
    args = get_arguments()

    paths = get_image_paths(args)
    pool = Pool(args.num_threads)

    for path in paths:
        pool.apply_async(worker, args=(path, args))

    pool.close()
    pool.join()
    print('All subprocesses done.')

if __name__ == "__main__":
    main()