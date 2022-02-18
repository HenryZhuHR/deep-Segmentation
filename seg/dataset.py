import os
import numpy
from PIL import Image
from torch.utils import data
from torchvision import transforms


class VOCSeg(data.Dataset):
    """
        - `root` : `<parent_path>/VOC2012`
    """

    def __init__(
        self,
        root: str = '~/datasets/VOC2012',
        train: bool = True,
        transform=None
    ) -> None:
        super().__init__()
        self.transform = transform
        with open(os.path.join(root, 'ImageSets/Segmentation/%s.txt' % 'train'if train else 'val'), 'r') as f:
            data_file_list = f.read().splitlines()
        self.data_len = len(data_file_list)
        self.dataset = []
        for file in data_file_list:
            image_file = os.path.join(root, 'JPEGImages/%s.jpg' % file)
            label_file = os.path.join(root, 'SegmentationClass/%s.png' % file)
            self.dataset.append([image_file, label_file])

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image = Image.open(self.dataset[index][0])
        label = Image.open(self.dataset[index][1])
        image = numpy.array(image, dtype=numpy.uint8)
        label = numpy.array(label, dtype=numpy.int32)
        if self.transform:
            return self.transform(image), self.transform(label)
        else:
            return image, label


if __name__ == '__main__':
    trainset = VOCSeg(
        root='%s/datasets/VOC2012' % ('E:'),
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    import tqdm
    pbar=tqdm.tqdm(trainset)
    for i,(image, label) in enumerate(pbar):
        pbar.set_description("%d: %s %s"%(i,image.size(), label.size()))
