
import os
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self,root,) -> None:
        super().__init__()
        for dir in os.listdir(root):
            print(dir)

def main():
    train_set=SegmentationDataset(root='data/train')

if __name__ == '__main__':
    main()
