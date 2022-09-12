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
        self.DATAROOT: str = args.data_root
        self.DATASET: str = args.dataset
        self.NUM_CLASSES: str = args.num_classes

        # ====== Model ======
        self.MODEL: str = args.model
        self.SEPARABLE_CONV: str = args.separable_conv
        self.LOSS_FUNCTION: str = args.loss_function
        
        # ====== Train ======
        self.DEVICE: str = torch.device(args.device)
        self.MAX_EPOCH: int = args.max_epoch
        self.LR: float = args.lr
        self.LR_SCHEDULE: str = args.lr_schedule
        self.STEP_SIZE: int = args.step_size
        self.WEIGHT_DECAY: float = args.weight_decay
        

    def init_args(self):
        parser = argparse.ArgumentParser()
        # ====== Dataset ======
        parser.add_argument("--data_root", type=str, default='./datasets/data')
        parser.add_argument("--dataset", type=str, default='voc',
                            choices=['voc', 'cityscapes', 'custom'],)
        parser.add_argument("--num_classes", type=int, default=21)

        # ====== Model ======
        parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50')
        parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
        parser.add_argument("--loss_function", type=str, default='focal_loss')
        
        # ====== Train ======
        parser.add_argument("--device", type=str, default='cpu')
        parser.add_argument("--max_epoch", type=int, default=30e3)
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--lr_schedule", type=str, default='poly', 
        choices=['poly', 'step'])
        parser.add_argument("--step_size", type=int, default=10000)
        parser.add_argument("--weight_decay", type=float, default=1e-4)

        # PASCAL VOC
        parser.add_argument("--year", type=str, default='2012',
                            choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

        # ====== Deeplab Model ======
        available_models = sorted(name for name in models.modeling.__dict__ if name.islower() and
                                  not (name.startswith("__") or name.startswith('_')) and callable(
            models.modeling.__dict__[name])
        )
        parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                            choices=available_models, help='model name')

        args = parser.parse_args()

        print(chr(128640), '\033[01;36m', args, '\033[0m')
        return args
