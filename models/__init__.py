from models.resnet import ResNet18, ResNet50, ResNet101, WideResNet50_2, WideResNet101_2
from models.resnet_kaiming import resnet20, resnet32, resnet32_double
from models.mobilenet import MobileNetV2
from models.frankle import FC, Conv2, Conv4, Conv4Normal, Conv6, Conv4Wide, Conv8, Conv6Wide
from models.vgg import vgg16

#### TODO: delete below ones (merge with above code)
from models.resnet_cifar import cResNet18, cResNet50 
from models.resnet_tiny import TinyResNet18 

__all__ = [
    "vgg16",
    "ResNet18",
    "ResNet50",
    "ResNet101",
    "WideResNet50_2",
    "WideResNet101_2",
    "resnet20",
    "resnet32",
    "resnet32_double",
    "MobileNetV2",
    "FC",
    "Conv2",
    "Conv4",
    "Conv4Normal",
    "Conv6",
    "Conv4Wide",
    "Conv8",
    "Conv6Wide",
    "cResNet18",
    "cResNet50",
    "TinyResNet18",
]
