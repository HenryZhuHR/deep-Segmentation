

from utils.args import ARGSer
from models import get_model


def print_train_info(ARGS: ARGSer):
    print('Device', ARGS.DEVICE)


def main():
    ARGS = ARGSer()
    model = get_model(ARGS.MODEL_ARCH, num_classes=ARGS.NUM_CLASSES,
                      pretrained_backbone=True)


if __name__ == '__main__':
    main()
