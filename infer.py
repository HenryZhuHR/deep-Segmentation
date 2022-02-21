# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py

import os
import argparse
import tqdm
import collections
import numpy
from PIL import Image
import torch
from torch import Tensor
from torch import cuda
from torch import nn
from torch.utils import data
from torchvision import models
from torchvision import transforms


FILE = './images/2007_000039'
image_path = FILE+'.jpg'
label_path = FILE+'.png'
mask_threshold=0.3


sem_classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

if __name__ == '__main__':
    device = 'cuda'if cuda.is_available()else 'cpu'

    # Data
    image = Image.open(image_path)
    input = Tensor(transform(image))
    input = input.unsqueeze(0)

    # Model
    model = models.segmentation.fcn_resnet50(pretrained=True, num_classes=21)
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        out_odict: collections.OrderedDict = model(input.to(device))
    output: Tensor = out_odict['out']

    masks: Tensor = nn.Softmax(dim=1)(output)
    print(masks.size())

    os.makedirs('results', exist_ok=True)
    for cls in sem_classes[1:]:
        for image_idx in range(masks.size(0)):
            mask_ts = masks[image_idx, sem_class_to_idx[cls], :, :]
            mask:numpy.ndarray = mask_ts.cpu().numpy()
            mask = mask > mask_threshold
            mask_pil = Image.fromarray(mask.astype(numpy.uint8) * 255)
            mask_pil.save(f'results/{cls}.png')
