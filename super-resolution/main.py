import argparse
import torch
import logging
import signal
import sys
import torch.backends.cudnn as cudnn
from trainer import Trainer
from datetime import datetime
from os import path
from utils import misc
from random import randint

# torch.autograd.set_detect_anomaly(True)

def get_arguments():
    parser = argparse.ArgumentParser(description='super-resolution')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-ids', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3')
    parser.add_argument('--d-model', default='d_srgan', help='discriminator architecture (default: srgan)')
    parser.add_argument('--g-model', default='g_srgan', help='generator architecture (default: srgan)')
    parser.add_argument('--model-config', default='', help='additional architecture configuration')
    parser.add_argument('--dis-to-load', default='', help='resume training from file (default: None)')
    parser.add_argument('--gen-to-load', default='', help='resume training from file (default: None)')
    parser.add_argument('--root', default='', help='root dataset folder')
    parser.add_argument('--scale', default=4, type=float, help='super-resolution scale (default: 4)')
    parser.add_argument('--crop-size', default=40, type=int, help='low resolution cropping size (default: 40)')
    parser.add_argument('--max-size', default=None, type=int, help='validation set max-size (default: None)')
    parser.add_argument('--num-workers', default=2, type=int, help='number of workers (default: 2)')
    parser.add_argument('--batch-size', default=16, type=int, help='batch-size (default: 16)')
    parser.add_argument('--epochs', default=1000, type=int, help='epochs (default: 1000)')
    parser.add_argument('--lr', default=2e-4, type=float, help='lr (default: 2e-4)')
    parser.add_argument('--gen-betas', default=[0.9, 0.99], nargs=2, type=float, help='scheduler gamma (default: 0.9, 0.99)')
    parser.add_argument('--dis-betas', default=[0.9, 0.99], nargs=2, type=float, help='scheduler gamma (default: 0.9, 0.99)')
    parser.add_argument('--num-critic', default=1, type=int, help='critic iterations (default: 1)')
    parser.add_argument('--wgan', default=False, action='store_true', help='critic wgan loss (default: false)')
    parser.add_argument('--relativistic', default=False, action='store_true', help='relativistic wgan loss (default: false)')
    parser.add_argument('--step-size', default=300, type=int, help='scheduler step size (default: 300)')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler gamma (default: 0.5)')
    parser.add_argument('--penalty-weight', default=0, type=float, help='gradient penalty weight (default: 0)')
    parser.add_argument('--range-weight', default=0, type=float, help='pixel-weight (default: 0)')
    parser.add_argument('--reconstruction-weight', default=1.0, type=float, help='reconstruction-weight (default: 1.0)')
    parser.add_argument('--perceptual-weight', default=0, type=float, help='perceptual-weight (default: 0)')
    parser.add_argument('--adversarial-weight', default=0.01, type=float, help='adversarial-weight (default: 0.01)')
    parser.add_argument('--textural-weight', default=0, type=float, help='textural-weight (default: 0)')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (default: random)')
    parser.add_argument('--print-every', default=20, type=int, help='print-every (default: 20)')
    parser.add_argument('--eval-every', default=50, type=int, help='eval-every (default: 50)')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
    parser.add_argument('--evaluation', default=False, action='store_true', help='evaluate a model (default: false)')
    parser.add_argument('--use-tb', default=False, action='store_true', help='use tensorboardx (default: false)')
    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save == '':
        args.save = time_stamp
    args.save_path = path.join(args.results_dir, args.save)
    if args.seed == -1:
        args.seed = randint(0, 12345)
    return args

def main():
    # arguments
    args = get_arguments()

    torch.manual_seed(args.seed)
    
    # cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # set logs
    misc.mkdir(args.save_path)
    misc.mkdir(path.join(args.save_path, 'images'))
    misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # print logs
    logging.info(args)

    # trainer
    trainer = Trainer(args)

    if args.evaluation:
        trainer.eval()
    else:
        trainer.train()

if __name__ == '__main__':
    # enables a ctrl-c without triggering errors
    # signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    main()