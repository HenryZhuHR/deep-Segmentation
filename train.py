
import torch
from torch import optim


import utils
from utils.args import ARGSer

from models import get_model,convert_to_separable_conv
from utils import set_bn_momentum

from metrics import StreamSegMetrics

def print_train_info(ARGS: ARGSer):
    print('Device', ARGS.DEVICE)


def main():
    ARGS = ARGSer()
    model = get_model(ARGS.MODEL, num_classes=ARGS.NUM_CLASSES,
                      pretrained_backbone=True)
    print(model)
    if ARGS.SEPARABLE_CONV and ('plus' in ARGS.MODEL):
        convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

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
        scheduler = utils.PolyLR(optimizer, ARGS.total_itrs, power=0.9)
    elif ARGS.LR_SCHEDULE == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=ARGS.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

if __name__ == '__main__':
    main()
