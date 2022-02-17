import os
import tqdm
from PIL import Image


def read_voc(root, max_num=None):

    txt_fname = '%s/ImageSets/Segmentation/%s.txt' % (root, 'train')

    with open(txt_fname, 'r') as f:
        file_name_list = f.read().split()  # 拆分成一个个名字组成list
    if max_num is not None:
        file_name_list = file_name_list[:min(max_num, len(images))]

    images = []
    labels = []
    pbar = tqdm.tqdm(file_name_list)
    for file_name in pbar:
        # 读入数据并且转为RGB的 PIL image
        images.append(
            Image.open('%s/JPEGImages/%s.jpg' %
                       (root, file_name)).convert("RGB")
        )
        labels.append(
            Image.open('%s/SegmentationClass/%s.png' %
                       (root, file_name)).convert("RGB")
        )
        pbar.set_description('[%s]' % file_name)
    return images, labels  # PIL image 0-255


read_voc(root='%s/datasets/VOC2012' % ('E:'))
