# local and global alignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from fvcore.nn.focal_loss import sigmoid_focal_loss_jit
from .build import DA_HEAD_REGISTRY
from ..layers.gradient_scalar_layer import GradientScalarLayer

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class GlobalAlignmentHead(nn.Module):
  output_channel = 512
  def __init__(self, context=False):
    super(GlobalAlignmentHead, self).__init__()
    self.conv1 = conv3x3(2048, 1024, stride=2)
    self.bn1 = nn.BatchNorm2d(1024)
    self.conv2 = conv3x3(1024, self.output_channel, stride=2)
    self.bn2 = nn.BatchNorm2d(self.output_channel)
    self.conv3 = conv3x3(self.output_channel, self.output_channel, stride=2)
    self.bn3 = nn.BatchNorm2d(self.output_channel)
    self.fc = nn.Linear(self.output_channel, 2)
    self.context = context
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
    x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
    x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
    x = F.avg_pool2d(x, (x.size(2),x.size(3)))
    x = x.view(-1, self.output_channel)
    if self.context:
      feat = x
    x = self.fc(x)
    if self.context:
      return x, feat
    else:
      return x


class LocalAlignmentHead(nn.Module):
  def __init__(self, context=False):
    super(LocalAlignmentHead, self).__init__()
    self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
    self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False)
    self.context = context
    self._init_weights()

  def _init_weights(self):
    def normal_init(m, mean, stddev):
      """
      weight initalizer: random normal.
      """
      # x is a parameter
      m.weight.data.normal_(mean, stddev)
      # m.bias.data.zero_()

    normal_init(self.conv1, 0, 0.01)
    normal_init(self.conv2, 0, 0.01)
    normal_init(self.conv3, 0, 0.01)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    if self.context:
      feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
      x = self.conv3(x)
      return x, feat
    else:
      x = self.conv3(x)
      return x


@DA_HEAD_REGISTRY.register()
class AlignmentHead(nn.Module):

  @configurable
  def __init__(self, *, local_alignment_on, global_alignment_on, gamma=5.0):
    # define network structure
    super().__init__()
    if local_alignment_on:
      self.localhead = LocalAlignmentHead(context=True)
      self.grl_localhead = GradientScalarLayer(-1.0)
    else:
      self.localhead = None
    if global_alignment_on:
      self.globalhead = GlobalAlignmentHead(context=True)
      self.grl_globalhead = GradientScalarLayer(-1.0)
    else:
      self.globalhead = None
    self.gamma = gamma

  @classmethod
  def from_config(cls, cfg):
    assert cfg.MODEL.DA_HEADS.LOCAL_ALIGNMENT_ON or cfg.MODEL.DA_HEADS.GLOBAL_ALIGNMENT_ON, 'domain adapatation head must have one alignment head (local or global) at least'
    return {
      'gamma': cfg.MODEL.DA_HEADS.GAMMA,
      'local_alignment_on': cfg.MODEL.DA_HEADS.LOCAL_ALIGNMENT_ON,
      'global_alignment_on': cfg.MODEL.DA_HEADS.GLOBAL_ALIGNMENT_ON,
    }
  """
  def forward(self, inputs):
    '''
    inputs: 
      dict[str->Tensor], local_head_feature:Tensor, global_head_feature:Tensor, feature domain:str
    outputs:
      training:
        feature_dict, loss_dict
      inference:
        feature_dict
    '''
    feat_local = inputs['local_head_feature']
    feat_global = inputs['global_head_feature']
    feat_domain = inputs['feature domain']

    reg_local_feat = None
    reg_global_feat = None
    loss = {}

    if self.localhead:
      # localhead branch
      _, reg_local_feat = self.localhead(feat_local.detach())
    if self.globalhead:
      # globalhead branch
      _, reg_global_feat = self.globalhead(feat_global.detach())

    if self.training:
      if self.localhead:
        feat_2d, _ = self.localhead(self.grl_localhead(feat_local))
        # local alignment, gan loss, l2-norm
        if feat_domain == 'source':
          domain_loss_local = 0.5 * torch.mean(torch.sigmoid(feat_2d) ** 2)
        elif feat_domain =='target':
          domain_loss_local = 0.5 * torch.mean(torch.sigmoid(1 - feat_2d) ** 2)
        loss.update({'loss_local_alignment': domain_loss_local})

      if self.globalhead:
        feat_value, _ = self.globalhead(self.grl_globalhead(feat_global))
        if feat_domain == 'source':
          domain_label = torch.ones_like(feat_value, requires_grad=True, device=feat_value.device)
        elif feat_domain == 'target':
          domain_label = torch.zeros_like(feat_value, requires_grad=True, device=feat_value.device)
        # global alignment, focal loss
        focal_loss_global = sigmoid_focal_loss_jit(feat_value, domain_label, gamma=self.gamma, reduction='mean')
        loss.update({'loss_global_alignment': focal_loss_global})

      return {'local_head_feature': reg_local_feat, 'global_head_feature': reg_global_feat}, loss

    else:
      return {'local_head_feature': reg_local_feat, 'global_head_feature': reg_global_feat}


def build_da_heads(cfg):
    return AlignmentHead(cfg)
"""

def forward(self, inputs):
    '''
    inputs: 
      dict[str->Tensor], local_head_feature:Tensor, global_head_feature:Tensor, feature domain:str
    outputs:
      training:
        feature_dict, loss_dict
      inference:
        feature_dict
    '''
    feat_local = inputs['local_head_feature']
    feat_global = inputs['global_head_feature']
    feat_domain = inputs['feature domain']

    reg_local_feat = None
    reg_global_feat = None
    loss = {}

    if self.localhead:
      # localhead branch
      _, reg_local_feat = self.localhead(feat_local.detach())
    if self.globalhead:
      # globalhead branch
      _, reg_global_feat = self.globalhead(feat_global.detach())

    if self.training:
      if self.localhead:
        feat_2d, _ = self.localhead(self.grl_localhead(feat_local))
        # local alignment, gan loss, l2-norm
        if feat_domain == 'source':
          domain_loss_local = 0.5 * torch.mean(torch.sigmoid(feat_2d) ** 2)
        elif feat_domain =='target':
          domain_loss_local = 0.5 * torch.mean(torch.sigmoid(1 - feat_2d) ** 2)
        loss.update({'loss_local_alignment': domain_loss_local})

      if self.globalhead:
        feat_value, _ = self.globalhead(self.grl_globalhead(feat_global))
        if feat_domain == 'source':
          domain_label = torch.ones_like(feat_value, requires_grad=True, device=feat_value.device)
        elif feat_domain == 'target':
          domain_label = torch.zeros_like(feat_value, requires_grad=True, device=feat_value.device)
        # global alignment, focal loss
        focal_loss_global = sigmoid_focal_loss_jit(feat_value, domain_label, gamma=self.gamma, reduction='mean')
        loss.update({'loss_global_alignment': focal_loss_global})

      return {'local_head_feature': reg_local_feat, 'global_head_feature': reg_global_feat}, loss

    else:
      return {'local_head_feature': reg_local_feat, 'global_head_feature': reg_global_feat}


def build_da_heads(cfg):
    return AlignmentHead(cfg)
