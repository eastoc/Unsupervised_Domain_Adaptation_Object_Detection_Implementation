# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import ResLayer
import torch.nn.functional as F

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight * grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class LocalAlignmentHead(BaseModule):
    
    def __init__(self, in_channel, context=False):
        super(LocalAlignmentHead, self).__init__()
        self.output_channel = 512
        # Tensor: [region_proposals, 2048, 7, 7]
        # input channel = 2048
        self.grl = GradientScalarLayer(weight=-1.0)
        self.conv1 = conv3x3(in_channel, 1024, stride=2)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = conv3x3(1024, self.output_channel, stride=2)
        self.bn2 = nn.BatchNorm2d(self.output_channel)
        self.conv3 = conv3x3(self.output_channel, self.output_channel, stride=2)
        self.bn3 = nn.BatchNorm2d(self.output_channel)
        self.fc = nn.Linear(self.output_channel, 2)
        self.context = context
        #self.softmax = nn.Softmax(dim=1)
        #self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = []
        x = self.grl(x)
    
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2),x.size(3)))
        x = x.view(-1, self.output_channel)
        
        if self.context:
            feat = x
        out = self.fc(x)
        out = torch.sigmoid(out)
        #out = self.softmax(out)
    
        '''
        if self.context:
            return out, feat
        else:
            return out
        '''
        return out
        
    def _init_weights(self):
        def normal_init(m, mean, stddev):
            """
            weight initalizer: random normal.
            """
            # x is a parameter
            m.weight.data.normal_(mean, stddev)
            # m.bias.data.zero_()

        normal_init(self.conv1, 0, 1)
        normal_init(self.conv2, 0, 1)
        normal_init(self.conv3, 0, 1)