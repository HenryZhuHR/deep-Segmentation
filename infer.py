

import os
import argparse
import tqdm
from PIL import Image
import torch
from torch import Tensor
from torch import cuda
from torch.utils import data
from torchvision import models
from torchvision import transforms


FILE='./images/2007_000032'
image_path=FILE+'.jpg'
label_path=FILE+'.png'

if __name__ == '__main__':
    device='cuda'if cuda.is_available()else 'cpu'

    # Data
    image=Image.open(image_path)
    input=Tensor(transforms.ToTensor()(image)).unsqueeze(0).to(device)
    print(input.size())

    # Model
    model=models.segmentation.fcn_resnet50(pretrained=True,num_classes=21)
    model.to(device)
    model.eval()

    # Inference
    with torch.no_grad():
        output:Tensor=model(input)
        print(output.size())
    output=output.squeeze(0).cpu().numpy()
    output=output.argmax(axis=0)




   