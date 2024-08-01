import torch
import torch.nn as nn

from utils import Normalize, weights_init_kaiming, weights_init_classifier, GeMP
from ResNet import resnet50
from loss import MMDLoss
import torch.nn.functional as F
from demo import DouHTansFeat 
from attention import PEBlock
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
        x = self.visible.maxpool(x)  #x.Size([32, 64, 72, 36])
        x = self.visible.layer1(x)  #torch.Size([32, 256, 72, 36])
        x = self.visible.layer2(x)  #torch.Size([32, 512, 36, 18])
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
        self.pe = PEBlock(1024)
       

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
            # MMD = 0.85* mmd + 0.15* MMD
            feat = self.bottleneck1(x_tran_pool)
            cls_score = self.classifier1(feat)


            return {
                'x_pool':[x_pool],
                'x_after_BN':x_after_BN,
                'mmd':mmd,
                'cls_id': x_cls,
                'tranx_pool':feat,
                'tran_cls':cls_score,
                
            }

        else:
            x = self.base(x)
            x = self.relu(x)
            #x_tran = self.transform(x)
            b, c, h, w = x.shape
            x = x.view(b, c, h * w)
            x_pool = self.pool(x)
            feat_afer_BN = self.bottleneck(x_pool)

        return F.normalize(x_pool, p=2.0, dim=1), F.normalize(feat_afer_BN, p=2.0, dim=1)


if __name__ == '__main__':
    net = embed_net(395)
    input = torch.randn(1, 3, 288, 144)
    out = net(input, input, modal=0)
    input1 = (torch.randn(1, 3, 288, 144), torch.randn(1, 3, 288, 144), 1)
    flops, params = profile(net, inputs=(input,input))
    flops, params = clever_format([flops, params], "%.3f")
    print("training phase...")
    print("params: ", params)
    print("flops: ", flops)
    print("*"*30)
    print(parameter_count_table(net))


    ######################################################################

    net.eval()
    flops, params = profile(net, inputs=(input,input, 1))
    flops, params = clever_format([flops, params],"%.3f")
    print("testing phase...")
    print("params: ", params)
    print("flops: ", flops)
    flops = FlopCountAnalysis(net, input1)
    print("FLOPs: ", flops.total() / 10e8, "G")






























# import torch
# import torch.nn as nn

# from utils import Normalize, weights_init_kaiming, weights_init_classifier, GeMP
# from ResNet import resnet50
# from loss import MMDLoss
# import torch.nn.functional as F
# from demo import DouHTansFeat


# class visible_module(nn.Module):
#     def __init__(self, ):
#         super(visible_module, self).__init__()

#         model_v = resnet50(pretrained=True,last_conv_stride=1,last_conv_dilation=1)
#         self.visible = model_v

#     def forward(self, x):
#         x = self.visible.conv1(x)
#         x = self.visible.bn1(x)
#         x = self.visible.relu(x)
#         x = self.visible.maxpool(x)
#         x = self.visible.layer1(x)
#         x = self.visible.layer2(x)
#         x1 = self.visible.layer4(self.visible.layer3(x)) #torch.Size([64, 2048, 9, 5])
#         return {'input1':x, 'output1':x1}


# class thermal_module(nn.Module):
#     def __init__(self, ):
#         super(thermal_module, self).__init__()

#         model_t = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)
#         self.thermal = model_t

#     def forward(self, x):
#         x = self.thermal.conv1(x)
#         x = self.thermal.bn1(x)
#         x = self.thermal.relu(x)
#         x = self.thermal.maxpool(x)
#         x = self.thermal.layer1(x)
#         x = self.thermal.layer2(x)#[32,512, 36, 18]
#         x2 = self.thermal.layer4(self.thermal.layer3(x))#torch.Size([32, 2048, 18, 9])
        
#         return {'input2':x, 'output2':x2}


# class base_resnet(nn.Module):
#     def __init__(self):
#         super(base_resnet, self).__init__()

#         base = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)
#         self.base = base
       

#     def forward(self, x):
        
#         x = self.base.layer3(x)
#         x = self.base.layer4(x)
#         return x

    
    
# class embed_net(nn.Module):
#     def __init__(self, class_num):

#         super(embed_net, self).__init__()
#         self.visible = visible_module()
#         self.thermal = thermal_module()
#         self.base = base_resnet()
#         self.transform  = DouHTansFeat()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.pool = GeMP()
#         self.relu = nn.ReLU()
#         self.MMD = MMDLoss()
#         self.bottleneck = nn.BatchNorm1d(2048)
#         self.bottleneck.apply(weights_init_kaiming)
#         self.bottleneck1 = nn.BatchNorm1d(2048)
#         self.bottleneck1.apply(weights_init_kaiming)
#         self.classifier = nn.Linear(2048, class_num,bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.classifier1 = nn.Linear(2048, class_num,bias=False)
#         self.classifier1.apply(weights_init_classifier)
       
        

#     def forward(self, x_v, x_t, modal=0):
#         if modal == 0:
#             x_v = self.visible(x_v)
#             x_t = self.thermal(x_t)
#             x = torch.cat([x_v['input1'], x_t['input2']], dim=0)
#             spec_v = x_v['output1']
#             spec_t = x_t['output2']
#             x = self.base(x)
#             del x_v, x_t
            
#         if modal == 1:
#             x_v = self.visible(x_v)
#             x = x_v['input1']
#             del x_v
#         if modal == 2:
#             x_t = self.thermal(x_t)
#             x = x_t['input2']
#             del x_t

#         if self.training:
#             """
#             sepc_v:[32, 2048, h, w]
#             spec_t:[32, 2048, h, w]
#             x:[64, 2048, h, w]
#             """
            
#             x_tran = self.transform(x, spec_v,spec_t) 
#             b, c, h, w = x.shape
#             x = self.relu(x)
#             x = x.view(b, c, h * w)

#             x_pool = self.pool(x)
#             mmd = self.MMD(F.normalize(x_pool[:b // 2], p=2, dim=1), F.normalize(x_pool[b // 2:], p=2, dim=1))
           
            
#             x_after_BN = self.bottleneck(x_pool)
            
#             x_cls = self.classifier(x_after_BN)
            
#             x_tran = self.relu(x_tran)
#             B,C,H,W = x_tran.shape
#             x_tran = x_tran.view(B,C,H*W)
            
#             x_tran_pool = self.pool(x_tran)
       
#             MMD = self.MMD(F.normalize(x_tran_pool[:B // 2], p=2, dim=1), F.normalize(x_tran_pool[B // 2:], p=2, dim=1))
#             MMD = 0.85* mmd + 0.15* MMD
#             feat = self.bottleneck1(x_tran_pool)
#             cls_score = self.classifier1(feat)


#             return {
#                 'x_pool':[x_pool],
#                 'x_after_BN':x_after_BN,
#                 'mmd':mmd,
#                 'cls_id': x_cls,
#                 'tranx_pool':feat,
#                 'tran_cls':cls_score,
                
#             }

#         else:
#             x = self.base(x)
#             x = self.relu(x)
#             # x_tran = self.transform(x)
#             b, c, h, w = x.shape
#             x = x.view(b, c, h * w)
#             x_pool = self.pool(x)
#             feat_afer_BN = self.bottleneck(x_pool)
            
           
#             #x_tran = self.relu(x_tran)
# #             b, c, h, w = x_tran.shape
            
# #             x_tran = x_tran.view(b,c, h*w)
# #             x_tran_pool = self.pool(x_tran)
            
# #             feat = self.bottleneck(x_tran_pool)

#         return F.normalize(x_pool, p=2.0, dim=1), F.normalize(feat_afer_BN, p=2.0, dim=1)







# """
# net = embed_net()
# res = net(x1, x2)
# loss_tri = tri(res['x_pool'])
# """



# if __name__ == '__main__':
#     net = embed_net(395)
#     x = torch.randn(32, 3, 288, 144)
#     y = net(x,x)
#     print(y)

# if __name__ == '__main__':
#     net = embed_net(395)
#     x = torch.randn(48, 3, 288, 144)
#     y = net(x,x)
#     # print(y)

