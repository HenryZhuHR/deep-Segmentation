

import os
import argparse
from torch.utils import data
from torchvision import models
from torchvision import transforms

from seg import VOCSeg

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='%s/datasets/VOC2012' % ('E:'))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

DATASET: str = args.dataset
BATCH_SIZE: int = args.batch_size
NUM_WORKERS: int = args.num_workers
TRANSFORM = {
    'train': transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),

    ]),
    'valid': transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
    ])
}
if __name__ == '__main__':
    # ====================================
    #   Load dataset
    # ====================================
    train_set = VOCSeg(
        root=DATASET,train=True,
        transform=TRANSFORM['train']
    )
    train_loader = data.DataLoader(
        dataset=train_set, shuffle=True,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    valid_set = VOCSeg(
        root=DATASET,train=False,
        transform=TRANSFORM['valid']
    )
    valid_loader = data.DataLoader(
        dataset=valid_set, shuffle=False,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    from seg.fcn import FCN_8s
    # model=models.segmentation.fcn_resnet50
    model=FCN_8s(n_class=21)