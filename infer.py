
import os
import time
import tqdm
import cv2
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch import optim
from torch.utils import data
from torchvision import transforms

import pandas as pd
import utils
from utils.args import ARGSer

from models import get_model, convert_to_separable_conv
from utils import set_bn_momentum
from datasets import ext_transforms as et
from datasets .custom import SegmentationDataset
from metrics import StreamSegMetrics


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
COLORS = (
    (0, 0, 0),
    (100, 149, 237),
    (0, 205, 102),
    (238, 99, 99),

)
# torch.set_printoptions(profile="full")


def main():
    ARGS = ARGSer()
    ARGS.MODEL = 'deeplabv3plus_resnet50'
    ARGS.NUM_CLASSES = 3
    ARGS.MODEL_WEIGHT = 'checkpoints/deeplabv3plus_resnet50-360.pt'
    ARGS.IMAGE_FILE = '/home/ubuntu/datasets/forProject/gc10_mask_mship/test/JPEGImages/cg_0407.jpg'

    model = get_model(ARGS.MODEL, num_classes=ARGS.NUM_CLASSES,
                      pretrained_backbone=True)
    if ARGS.SEPARABLE_CONV and ('plus' in ARGS.MODEL):
        convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    assert ARGS.MODEL_WEIGHT is not None
    state_dict = torch.load(ARGS.MODEL_WEIGHT, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(ARGS.DEVICE)
    model.eval()

    img_src: np.ndarray = cv2.imread(ARGS.IMAGE_FILE)
    img: np.ndarray = cv2.resize(img_src, dsize=(299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.

    image = torch.from_numpy(img)
    image = image.to(ARGS.DEVICE, dtype=torch.float32)
    # image=image.unsqueeze(0)

    output: Tensor = model(image)
    prob: Tensor = output.detach().max(dim=1).values
    cls: Tensor = output.detach().max(dim=1).indices
    prob = prob.squeeze(0)
    cls = cls.squeeze(0)
    cls_np: np.ndarray=cls.numpy()

    mask_np: np.ndarray = np.zeros(img_src.shape, np.uint8)
    for i in range(cls_np.shape[0]):
        for j in range(cls_np.shape[1]):
            mask_np[i][j]=COLORS[cls_np[i][j]]

    result=cv2.addWeighted(img_src,0.5,mask_np,0.5,0)
    print(img_src.shape)
    print(mask_np.shape)
    cv2.imwrite('img-src.png',img_src)
    cv2.imwrite('img-mask.png',mask_np)
    cv2.imwrite('img-result.png',result)


if __name__ == '__main__':
    main()
