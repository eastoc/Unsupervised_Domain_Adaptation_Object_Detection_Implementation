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

class InstanceAlignmentHead(BaseModule):
    
    def __init__(self, context=False):
        super(InstanceAlignmentHead, self).__init__()
        # Tensor: [region_proposals, 1, 1024,]
        # input channel = 1024
        self.training = True
        self.grl = GradientScalarLayer(weight=-1.0)
        self.nlb = NonLocalBlock(1024)
        self.nlb._init_weights()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 2)
        self.context = context
        #self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        #print('x',x.size())
        x = self.grl(x)# [k, C]
        x = x.unsqueeze(dim=0)# [b, k, C]
        x = x.permute(0, 2, 1).contiguous()# [b, C, k]
        x = x.unsqueeze(dim=2)# [b, C, 1, k]
        x = self.nlb(x)
        x = x.permute(3,1,0,2).contiguous()# [k, C, b, 1]
        x = x.squeeze(dim=-1)
        x = x.squeeze(dim=-1)# [k, C]
        #print('out',x.size())
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        x = self.fc3(x)
        
        if self.context:
            feat = x
        
        out = torch.sigmoid(x)
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

        normal_init(self.fc1, 0, 0.01)
        normal_init(self.fc2, 0, 0.01)
        normal_init(self.fc3, 0, 0.05)

class InstanceAlignmentHead_DAF(BaseModule):
    
    def __init__(self, context=False):
        super(InstanceAlignmentHead_DAF, self).__init__()
        # Tensor: [region_proposals, 1, 1024,]
        # input channel = 1024
        self.training = True
        self.grl = GradientScalarLayer(weight=-1.0)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.context = context

    def forward(self, x):
        x = self.grl(x)
        
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        x = self.fc3(x)
        
        if self.context:
            feat = x
        
        out = torch.sigmoid(x)
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

        normal_init(self.fc1, 0, 0.01)
        normal_init(self.fc2, 0, 0.01)
        normal_init(self.fc3, 0, 0.01)

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
    
    def _init_weights(self):
        def normal_init(m, mean, stddev):
            """
            weight initalizer: random normal.
            """
            # x is a parameter
            m.weight.data.normal_(mean, stddev)
            # m.bias.data.zero_()

        normal_init(self.conv_phi, 0, 0.01)
        normal_init(self.conv_theta, 0, 0.01)
        normal_init(self.conv_g, 0, 0.01)
        normal_init(self.conv_mask, 0, 0.01)

    def forward(self, x):
        # [N, C, k, 1]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out
