import torch
import torch.nn as nn

from utils import Normalize, weights_init_kaiming, weights_init_classifier, GeMP
from ResNet import resnet50
from loss import MMDLoss
import torch.nn.functional as F
from attention import MCPblock, DouHTansFeat 
from thop import clever_format, profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class visible_module(nn.Module):
    def __init__(self):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,last_conv_stride=1,last_conv_dilation=1)
        self.visible = model_v
           
    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)  
        x = self.visible.layer1(x)  
        x = self.visible.layer2(x)  
        return x


class thermal_module(nn.Module):
    def __init__(self):
        super(thermal_module, self).__init__()
       
        model_t = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t      

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        return x


class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()

        base = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)
        self.base = base
        self.pe = MCPblock(1024)
       

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.pe(x)
        x = self.base.layer4(x)     
        return x

    
    
class embed_net(nn.Module):
    def __init__(self, class_num):
        super(embed_net, self).__init__()
        self.visible = visible_module()
        self.thermal = thermal_module()
        self.base = base_resnet()
        self.transform  = DouHTansFeat()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.pool = GeMP()
        self.relu = nn.ReLU()
        self.MMD = MMDLoss()
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck1 = nn.BatchNorm1d(2048)
        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier = nn.Linear(2048, class_num,bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier1 = nn.Linear(2048, class_num,bias=False)
        self.classifier1.apply(weights_init_classifier)
       
        

    def forward(self, x_v, x_t, modal=0):
        if modal == 0:
            x_v = self.visible(x_v)
            x_t = self.thermal(x_t)
            x = torch.cat((x_v, x_t), dim=0)      
        
        elif modal == 1:
            x = self.visible(x_v)
                        
        elif modal == 2:
            x = self.thermal(x_t)
        
           
        if self.training:
            x = self.base(x)
            x_tran = self.transform(x) 
            b, c, h, w = x.shape   
            x = self.relu(x)
            x = x.view(b, c, h * w)

            x_pool = self.pool(x)
            mmd = self.MMD(F.normalize(x_pool[:b // 2], p=2, dim=1), F.normalize(x_pool[b // 2:], p=2, dim=1))      
            x_after_BN = self.bottleneck(x_pool)     
            x_cls = self.classifier(x_after_BN)
            
            x_tran = self.relu(x_tran)
            B,C,H,W = x_tran.shape
            x_tran = x_tran.view(B,C,H*W)      
            x_tran_pool = self.pool(x_tran)
       
            MMD = self.MMD(F.normalize(x_tran_pool[:B // 2], p=2, dim=1), F.normalize(x_tran_pool[B // 2:], p=2, dim=1))
            feat = self.bottleneck1(x_tran_pool)
            cls_score = self.classifier1(feat)


            return {
                'x_pool':[x_pool],
                'x_after_BN':x_after_BN,
                'mmd':mmd,
                'cls_id': x_cls,
                
            }

        else:
            x = self.base(x)
            x = self.relu(x)
            b, c, h, w = x.shape
            x = x.view(b, c, h * w)
            x_pool = self.pool(x)
            feat_afer_BN = self.bottleneck(x_pool)

        return F.normalize(x_pool, p=2.0, dim=1), F.normalize(feat_afer_BN, p=2.0, dim=1)
