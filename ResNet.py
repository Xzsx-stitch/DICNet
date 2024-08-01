import torch.nn as nn
import math
import torch.utils as model
from torch.nn import init
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  """3x3 convolution with padding"""
  # original padding is 1; original dilation is 1
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
      in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride, dilation)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    # original padding is 1; original dilation is 1
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):

    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

class rnewBottleNeck(nn.Module):
   def __init__(self, out_planes=2048, class_num=395):
      super(rnewBottleNeck, self).__init__()
      self.num_classes = class_num
      self.gap = nn.AdaptiveAvgPool2d(1)
      self.num_features = out_planes
      self.feat_bn = nn.BatchNorm1d(self.num_features)
      self.feat_bn.bias.requires_grad_(False)
      init.constant_(self.feat_bn.weight, 1)
      init.constant_(self.feat_bn.bias, 0)
      self.part_detach = True

      self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
      init.normal_(self.classifier.weight, std=0.001)
      norm_layer = nn.BatchNorm2d
      block = Bottleneck
      planes = 512
      self.planes = planes
      downsample = nn.Sequential(
        conv1x1(out_planes, planes * block.expansion),
        norm_layer(planes * block.expansion),
      )
      self.part_bottleneck = block(
        out_planes, planes, downsample=downsample, norm_layer=norm_layer
      )

      self.part_num_features = planes * block.expansion
      self.part_pool = nn.AdaptiveAvgPool2d((2, 1))

      self.partup_feat_bn = nn.BatchNorm1d(self.part_num_features)
      self.partup_feat_bn.bias.requires_grad_(False)
      init.constant_(self.partup_feat_bn.weight, 1)
      init.constant_(self.partup_feat_bn.bias, 0)

      self.partdown_feat_bn = nn.BatchNorm1d(self.part_num_features)
      self.partdown_feat_bn.bias.requires_grad_(False)
      init.constant_(self.partdown_feat_bn.weight, 1)
      init.constant_(self.partdown_feat_bn.bias, 0)

      self.classifier_partup = nn.Linear(self.part_num_features, self.num_classes, bias=False)
      init.normal_(self.classifier_partup.weight, std=0.001)
      self.classifier_partdown = nn.Linear(self.part_num_features, self.num_classes, bias=False)
      init.normal_(self.classifier_partdown.weight, std=0.001)

      if not self.pretrained:
        self.reset_params()

   def forward(self, x, finetune=False):
     featuremap = ResNet(x)

     x = self.gap(featuremap)
     x = x.view(x.size(0), -1)

     bn_x = self.feat_bn(x)

     if self.part_detach:
       part_x = self.part_bottleneck(featuremap.detach())
     else:
       part_x = self.part_bottleneck(featuremap)

     part_x = self.part_pool(part_x)
     part_up = part_x[:, :, 0, :]
     part_up = part_up.view(part_up.size(0), -1)
     bn_part_up = self.partup_feat_bn(part_up)

     part_down = part_x[:, :, 1, :]
     part_down = part_down.view(part_down.size(0), -1)
     bn_part_down = self.partdown_feat_bn(part_down)

     if self.training is False and finetune is False:
       bn_x = F.normalize(bn_x)
       return [bn_x]

     prob = self.classifier(bn_x)
     prob_part_up = self.classifier_partup(bn_part_up)
     prob_part_down = self.classifier_partdown(bn_part_down)

     if finetune is True:
       bn_x = F.normalize(bn_x)
       bn_part_up = F.normalize(bn_part_up)
       bn_part_down = F.normalize(bn_part_down)
       return [x, part_up, part_down], [bn_x, bn_part_up, bn_part_down], [prob, prob_part_up, prob_part_down]
     else:
       return [x, part_up, part_down], [prob, prob_part_up, prob_part_down]


def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  # for key, value in state_dict.items():
  for key, value in list(state_dict.items()):
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
  return model


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    # model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet101'])))
  return model


def resnet152(pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet152'])))
  return model
