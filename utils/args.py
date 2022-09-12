from email.policy import default
import os
import argparse
from typing import List
import torch
import models

DEFAULT_DATAROOT = os.path.expandvars('$HOME/datasets')


class ARGSer:
    def __init__(self) -> None:
        args = self.init_args()
        self.args = args

        # ====== Dataset ======
        self.DATAROOT: str = os.path.expanduser(args.data_root)
        self.NUM_CLASSES: str = args.num_classes
        self.SIZE: str = args.size

        # ====== Model ======
        self.MODEL: str = args.model
        self.SEPARABLE_CONV: str = args.separable_conv
        self.LOSS_FUNCTION: str = args.loss_function

        # ====== Train ======
        self.DEVICE: str = torch.device(args.device)
        self.TRAIN_BATCH_SIZE: int = args.train_batch_size
        self.VALID_BATCH_SIZE: int = args.valid_batch_size
        self.NUM_WORKERS: int = args.num_workers
        self.EPOCHS: int = int(args.epochs)
        self.LR: float = args.lr
        self.LR_SCHEDULE: str = args.lr_schedule
        self.STEP_SIZE: int = args.step_size
        self.WEIGHT_DECAY: int = args.weight_decay

        self.SAVE_DIR: str = args.save_dir
        self.SAVE_NAME: str = args.save_name

    def init_args(self):
        parser = argparse.ArgumentParser()
        # ====== Dataset ======
        parser.add_argument('--data_root', type=str,
                            default=os.path.expanduser('~/datasets/gc10_mask_mship'))
        parser.add_argument('--num_classes', type=int, default=21)
        parser.add_argument('--size', type=int, default=299)

        # ====== Model ======
        parser.add_argument('--model', type=str,
                            default='deeplabv3plus_resnet50')
        parser.add_argument('--separable_conv', action='store_true', default=False,
                            help='apply separable conv to decoder and aspp')
        parser.add_argument('--loss_function', type=str, default='focal_loss')

        # ====== Train ======
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--train_batch_size', type=int, default=4)
        parser.add_argument('--valid_batch_size', type=int, default=4)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--epochs', type=int, default=30e3)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--lr_schedule', type=str, default='poly',
                            choices=['poly', 'step'])
        parser.add_argument('--step_size', type=int, default=10000)
        parser.add_argument('--weight_decay', type=float, default=1e-4)

        parser.add_argument('--save_dir', type=str, default='checkpoints')
        parser.add_argument('--save_name', type=str,
                            default='deeplabv3plus_resnet50')

        args = parser.parse_args()

        print(chr(128640), '\033[01;36m', args, '\033[0m')
        return args
