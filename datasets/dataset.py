import os
from typing import List, Tuple
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from . import ext_transforms as et

train_transforms = et.ExtCompose([
    et.ExtResize(299),
    et.ExtRandomCrop(size=(299, 299)),
    et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    et.ExtRandomHorizontalFlip(),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

valid_transforms = et.ExtCompose([
    et.ExtResize(299),
    et.ExtToTensor(),    
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])


class SegmentationDataset(Dataset):
    def __init__(self,root:str,transforms: et.ExtCompose = train_transforms) -> None:
        super().__init__()
        
        assert os.path.exists(root), f"path '{root}' dose not exits."
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.npy_dir = os.path.join(root, 'SegmentationClass')
        
        self.transform = transforms
        self.image_list = [] # 图片列表
        self.label_list = [] # 标签列表
        
        for filename_image in os.listdir(self.img_dir):
            filename_npy = os.path.splitext(filename_image)[0] + '.npy'

            image_file_path = os.path.join(self.img_dir, filename_image)
            npy_file_path = os.path.join(self.npy_dir, filename_npy)

            if not os.path.exists(npy_file_path):
                raise FileNotFoundError(f'label of {image_file_path} -> {npy_file_path} not found')
            
            image_pil = Image.open(image_file_path) 
            label_np = np.load(npy_file_path)
            label_pil = Image.fromarray(label_np)

            self.image_list.append(image_pil)
            self.label_list.append(label_pil)

    def __getitem__(self, index:int)->Tuple[Tensor,Tensor]:
        if self.transform is not None:
            image, label = self.transform(self.image_list[index], self.label_list[index])
        return image,label

    def __len__(self)->int:
        return len(self.image_list)

if __name__ == '__main__':
    train_set=SegmentationDataset(root='D:/projects/partsegvis/partsegvis/dataset/gc10_mask_mship/train', transforms = train_transforms)

    print('image tensor',train_set[0][0].size(),train_set[0][0])
    print('label tensor',train_set[0][1].size(),train_set[0][1])
