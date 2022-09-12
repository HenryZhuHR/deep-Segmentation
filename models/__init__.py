from typing import Dict
from .modeling import (
    DeepLabV3,

    # Deeplab v3
    deeplabv3_hrnetv2_48,
    deeplabv3_hrnetv2_32,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet,
    deeplabv3_xception,

    # Deeplab v3+
    deeplabv3plus_hrnetv2_48,
    deeplabv3plus_hrnetv2_32,
    deeplabv3plus_resnet50,
    deeplabv3plus_resnet101,
    deeplabv3plus_mobilenet,
    deeplabv3plus_xception,
)
from ._deeplab import convert_to_separable_conv

MODEL_LIST: Dict[str, DeepLabV3] = {
    # Deeplab v3
    'deeplabv3_hrnetv2_48': deeplabv3_hrnetv2_48,
    'deeplabv3_hrnetv2_32': deeplabv3_hrnetv2_32,
    'deeplabv3_resnet50': deeplabv3_resnet50,
    'deeplabv3_resnet101': deeplabv3_resnet101,
    'deeplabv3_mobilenet': deeplabv3_mobilenet,
    'deeplabv3_xception': deeplabv3_xception,

    # Deeplab v3+
    'deeplabv3plus_hrnetv2_48': deeplabv3plus_hrnetv2_48,
    'deeplabv3plus_hrnetv2_32': deeplabv3plus_hrnetv2_32,
    'deeplabv3plus_resnet50': deeplabv3plus_resnet50,
    'deeplabv3plus_resnet101': deeplabv3plus_resnet101,
    'deeplabv3plus_mobilenet': deeplabv3plus_mobilenet,
    'deeplabv3plus_xception': deeplabv3plus_xception,
}


def get_model(model_name: str, num_classes=21, output_stride=8, pretrained_backbone=True) -> DeepLabV3:
    if model_name in MODEL_LIST:
        model = MODEL_LIST[model_name]
        return model(num_classes=num_classes, output_stride=output_stride,
                     pretrained_backbone=pretrained_backbone)
    else:
        raise KeyError('model_name only available in', MODEL_LIST)
