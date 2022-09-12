from email.policy import default
import os
import argparse
from typing import List
import torch
import network

DEFAULT_DATAROOT = os.path.expandvars('$HOME/datasets')


class ARGSer:
    def __init__(self) -> None:
        args = self.init_args()
        self.args = args

        # ====== Dataset ======
        self.DATAROOT: str = args.data_root
        self.DATASET: str = args.dataset
        self.NUM_CLASSES: str = args.num_classes
        

    def init_args(self):
        parser = argparse.ArgumentParser()
        # ====== Dataset ======
        parser.add_argument("--data_root", type=str, default='./datasets/data')
        parser.add_argument("--dataset", type=str, default='voc',
                            choices=['voc', 'cityscapes', 'custom'],)
        parser.add_argument("--num_classes", type=int, default=21,)
        # PASCAL VOC
        parser.add_argument("--year", type=str, default='2012',
                            choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

        # ====== Deeplab Model ======
        available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and
                                  not (name.startswith("__") or name.startswith('_')) and callable(
            network.modeling.__dict__[name])
        )
        parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                            choices=available_models, help='model name')

        args = parser.parse_args()

        print(chr(128640), '\033[01;36m', args, '\033[0m')
        return args
