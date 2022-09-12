
import os
from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self,root:str,) -> None:
        super().__init__()
        for dir in os.listdir(root):
            print(dir)

    def __getitem__(self, index:int)->Tuple[Tensor,Tensor]:
        image:Tensor=None
        label:Tensor=None
        return image,label

    def __len__(self)->int:
        pass

if __name__ == '__main__':
    train_set=SegmentationDataset(root='D:/projects/partsegvis/partsegvis/dataset/gc10_mask_mship/train')

    print('image tensor',train_set[0][0])
    print('label tensor',train_set[0][1])
    print('ssd')