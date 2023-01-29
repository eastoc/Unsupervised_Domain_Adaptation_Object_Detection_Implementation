# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .resnet_da import ResNet_DA
from .resnet_da_daf_org import ResNet_DAF
from .resnet_da_v2 import ResNet_DA_v2
from .resnet_da_v3 import ResNet_DA_v3
from .resnet_cycada import ResNet_cycada
from .global_da import GlobalAlignmentHead_top
from .global_da import GlobalAlignmentHead_mid
from .global_da import GlobalAlignmentHead_bottom
from .resnet_da_cbam import ResNet_DA_CBAM
from .resnet_da_tri_att import ResNet_DA_Tri_Att
from .resnet_da_swda import ResNet_DA_SWDA
from .resnet_da_deep import ResNet_DA_Deep

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer', 'PyramidVisionTransformerV2',
    'ResNet_DA','ResNet_DA_v2','ResNet_DA_v3','ResNet_DA_CBAM','ResNet_cycada','ResNet_DA_Tri_Att',
    'GlobalAlignmentHead_top','GlobalAlignmentHead_mid','GlobalAlignmentHead_bottom','ResNet_DAF',
    'ResNet_DA_SWDA','ResNet_DA_Deep'
]
