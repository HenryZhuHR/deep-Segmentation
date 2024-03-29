
import os
import time
import tqdm

import torch
from torch import nn
from torch import Tensor
from torch import optim
from torch.utils import data
import pandas as pd
import utils
from utils.args import ARGSer

from models import get_model, convert_to_separable_conv
from utils import set_bn_momentum
from datasets import ext_transforms as et
from datasets .dataset import SegmentationDataset
from metrics import StreamSegMetrics


def print_train_info(ARGS: ARGSer):
    print('Device', ARGS.DEVICE)


def main():
    ARGS = ARGSer()

    CSV_FILE='%s/%s.csv'%(ARGS.SAVE_DIR,ARGS.SAVE_NAME)
    os.makedirs(ARGS.SAVE_DIR, exist_ok=True)
    open(CSV_FILE,'w').close()

    train_set, valid_set = get_dataset(ARGS)
    train_loader = data.DataLoader(train_set,  batch_size=ARGS.TRAIN_BATCH_SIZE,
                                   shuffle=True, num_workers=ARGS.NUM_WORKERS)
    valid_loader = data.DataLoader(valid_set,  batch_size=ARGS.VALID_BATCH_SIZE,
                                   shuffle=False, num_workers=ARGS.NUM_WORKERS)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (ARGS.DATAROOT, len(train_set), len(valid_set)))

    model = get_model(ARGS.MODEL, num_classes=ARGS.NUM_CLASSES,
                      pretrained_backbone=True)
    if ARGS.SEPARABLE_CONV and ('plus' in ARGS.MODEL):
        convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model.to(ARGS.DEVICE)

    # Set up metrics
    metrics = StreamSegMetrics(ARGS.NUM_CLASSES)

    # Set up optimizer
    optimizer = optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * ARGS.LR},
        {'params': model.classifier.parameters(), 'lr': ARGS.LR},
    ], lr=ARGS.LR, momentum=0.9, weight_decay=ARGS.WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    if ARGS.LR_SCHEDULE == 'poly':
        scheduler = utils.PolyLR(optimizer, ARGS.EPOCHS, power=0.9)
    elif ARGS.LR_SCHEDULE == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=ARGS.STEP_SIZE, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if ARGS.LOSS_FUNCTION == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif ARGS.LOSS_FUNCTION == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    criterion.to(ARGS.DEVICE)

    

    # ==========   Train Loop   ==========#
    train_frame = None
    for epoch in range(ARGS.EPOCHS):
        print('\033[32m', end='')  # 设置字体颜色
        print('[Epoch]%d/%d' % (epoch, ARGS.EPOCHS), end=' ')
        print('[Batch Size]%d/%d' %
              (ARGS.TRAIN_BATCH_SIZE, ARGS.VALID_BATCH_SIZE), end=' ')
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('[LR]%f' % (current_lr), end=' ')
        print('[%s]' % (ARGS.DEVICE), end='  ')
        print('\033[0m')

        epoch_start_time = time.time()

        model.train()
        epoch_num = 0
        epoch_loss = 0.
        pbar = tqdm.tqdm(train_loader)
        for images, labels in pbar:
            images: Tensor = images.to(ARGS.DEVICE, dtype=torch.float32)
            labels: Tensor = labels.to(ARGS.DEVICE, dtype=torch.long)

            epoch_num += images.size(0)
            optimizer.zero_grad()
            outputs:Tensor = model(images)
            loss = criterion.forward(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.detach().cpu().numpy()
            epoch_loss += batch_loss
            epoch_info = 'Train Loss:%.4f' % batch_loss
            pbar.set_description(epoch_info)
        scheduler.step()
        train_loss=epoch_loss/epoch_num

        model.eval()
        epoch_num = 0
        epoch_loss = 0.
        pbar = tqdm.tqdm(valid_loader)  # valid_loader
        metrics.reset()
        for images, labels in pbar:
            images: Tensor = images.to(ARGS.DEVICE)
            labels: Tensor = labels.to(ARGS.DEVICE)
            labels=labels.long()
            epoch_num += images.size(0)
            with torch.no_grad():
                outputs:Tensor = model(images)
                loss = criterion.forward(outputs, labels)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            batch_loss = loss.detach().cpu().numpy()
            epoch_loss += batch_loss
            epoch_info = 'Valid Loss:%.4f' % batch_loss
            pbar.set_description(epoch_info)
        """
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,    
        """
        score = metrics.get_results()
        valid_loss=epoch_loss/epoch_num

        results={
            'epoch':epoch,
            'train_loss':train_loss,
            'valid_loss':valid_loss,
        }
        for key,value in score.items():
            if key =='Class IoU':continue
            results[key]=value

        log_frame = pd.DataFrame(results, index=[epoch])
        print(log_frame)

        if train_frame is None:
            train_frame = log_frame
        else:
            train_frame = pd.concat(
                [train_frame, log_frame], ignore_index=True)
        train_frame.to_csv(CSV_FILE, index=False, header=True)

        save_model ='%s/%s-%d.pt'%(ARGS.SAVE_DIR, ARGS.SAVE_NAME,epoch)
        torch.save(model.state_dict(),save_model)



def get_dataset(ARGS: ARGSer):
    train_transforms = et.ExtCompose([
        et.ExtResize(size=ARGS.SIZE),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(ARGS.SIZE, ARGS.SIZE), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    train_set = SegmentationDataset(root=os.path.join(ARGS.DATAROOT, 'train'),
                                    transforms=train_transforms)
    valid_set = SegmentationDataset(root=os.path.join(ARGS.DATAROOT, 'valid'),
                                    transforms=val_transforms)

    return train_set, valid_set


if __name__ == '__main__':
    main()
