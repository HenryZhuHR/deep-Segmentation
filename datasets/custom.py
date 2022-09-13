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
    def __init__(self, root: str, transforms: et.ExtCompose = train_transforms) -> None:
        super().__init__()

        images_dir = os.path.join(root, 'JPEGImages')
        labels_dir = os.path.join(root, 'SegmentationClass')

        images_list: List[Tensor] = []
        labels_list: List[Tensor] = []

        for file in os.listdir(images_dir):
            file_name = os.path.splitext(file)[0]
            image_file = os.path.join(images_dir, file)
            label_file = os.path.join(labels_dir, '%s.npy' % file_name)
            if not os.path.exists(label_file):
                raise FileNotFoundError(
                    f'label of {image_file} -> {label_file} not found')

            # if os.path.exists(image_file):
            if os.path.exists(image_file) and os.path.exists(label_file):
                image_pil = Image.open(image_file)
                label_np: np.ndarray = np.load(label_file)
                label_pil=Image.fromarray(label_np)
                if transforms is not None:
                    image, label = transforms(image_pil, label_pil)
                    images_list.append(image)
                    labels_list.append(label)
        self.images_list = images_list
        self.labels_list = labels_list

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image: Tensor = self.images_list[index]
        label: Tensor = self.labels_list[index]
        return image, label

    def __len__(self) -> int:
        return len(self.images_list)


if __name__ == '__main__':
    train_set = SegmentationDataset(root=os.path.expanduser('~/datasets/gc10_mask_mship/train'), 
    transforms=train_transforms)

    print('image tensor', train_set[0][0].size(),train_set[0][0])
    print('label tensor', train_set[0][1].size(),train_set[0][1])
