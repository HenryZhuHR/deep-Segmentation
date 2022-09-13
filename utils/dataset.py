import os
import cv2
from typing import Tuple
from torchvision import transforms
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

transform = transforms.Compose([transforms.ToTensor(), # 将图像转化为Tensor并归一化
])

class SegmentationDataset(Dataset):
    def __init__(self,root:str,) -> None:
        super().__init__()
        assert os.path.exists(root), f"path '{root}' dose not exits."
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.npy_dir = os.path.join(root, 'SegmentationClass')
        self.transform = transform
        self.image_list = []
        self.data_list = [] # 数据列表
        for filename_image in os.listdir(self.img_dir):
            filename_npy = os.path.splitext(filename_image)[0] + '.npy'

            image_file_path = os.path.join(self.img_dir, filename_image)
            npy_file_path = os.path.join(self.npy_dir, filename_npy)

            assert os.path.exists(npy_file_path), f"path '{npy_file_path}' dose not exits."

            self.image_list.append(cv2.imread(image_file_path))
            self.data_list.append(np.load(npy_file_path))

    def __getitem__(self, index:int)->Tuple[Tensor,Tensor]:
        image:Tensor=None
        label:Tensor=None

        image = self.transform(self.image_list[index])  # torch.Size([3, 299, 299])
        label = self.transform(self.data_list[index])   # torch.Size([1, 299, 299])
        
        return image,label

    def __len__(self)->int:
        return len(self.data_list)

if __name__ == '__main__':
    train_set=SegmentationDataset(root='D:/projects/partsegvis/partsegvis/dataset/gc10_mask_mship/train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True)
    #print('image tensor',train_set[0][0])
    #print('label tensor',train_set[0][1])
