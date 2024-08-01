import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Function
import logging
from data_loader import SYSUData, RegDBData,LLCMData,TestData
from data_manager import *
import time
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import math
import torch.nn as nn
import random
import torch.nn.functional as F
from loss import OriTripletLoss, TripletLoss_WRT, PairCircleLoss, MMDLoss, KLDivLoss
from random_erasing import RandomErasing
# from corruptions import corruption_transform


# you can add your implemented metric loss here
def generate_metric_loss(args, **kwargs):
    metric_type = args.metric
    print('Generating Metric loss as ', metric_type)
    if metric_type == 'Triplet':
        # original triplet loss
        metric_loss = OriTripletLoss(**kwargs)
    elif metric_type == 'WRTriplet':
        # weight regulated triplet loss
        metric_loss = TripletLoss_WRT()
    elif metric_type == 'Circle':
        # paired circle loss for ReID
        metric_loss = PairCircleLoss(**kwargs)
    else:
        raise NotImplementedError('Not implemented metric loss {}'.format(metric_type))

    return metric_loss.cuda()


# you can add your implemented identity loss here
def generate_identity_loss(args, **kwargs):
    id_type = args.id
    print('Generating identity loss as ', id_type)
    if id_type == 'CrossEntropy':
        # cross-entropy loss with label_smoothing (optional)
        identity_loss = nn.CrossEntropyLoss(**kwargs)
    else:
        raise NotImplementedError('Not implemented identity loss {}'.format(id_type))
    return identity_loss


# you can add your implemented alignment loss here
def generate_alignment_loss(args, **kwargs):
    align_type = args.alignment
    print('Generating alignment loss as ', align_type)
    if align_type == 'MMD':
        # mmd without margin
        alignment_loss = MMDLoss(**kwargs)
    elif align_type == 'KLDIV':
        # KL-Divergence
        alignment_loss = KLDivLoss()
    else:
        raise NotImplementedError('Not implemented identity loss {}'.format(align_type))
    return alignment_loss


def prepare_optimizer(net,args):
    # it is not necessary to use Adam when you have AdamW :)
    opti_type = args.optimizer
    assert opti_type in ['AdamW','SGD']
    ignored_params = list(map(id, net.classifier.parameters())) \
                     + list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.base.base.layer3.parameters())) \
                     + list(map(id, net.base.base.layer4.parameters())) \
                     + list(map(id, net.base.pe.parameters())) \
                     + list(map(id, net.transform.parameters())) 



    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    if opti_type == 'AdamW':
        optimizer = optim.AdamW([{'params': base_params, 'lr': 0.1 * args.lr,},
                                 {'params': net.bottleneck.parameters(), 'lr': args.lr},
                                 {'params': net.classifier.parameters(), 'lr': args.lr},],weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * args.lr,},
                                 {'params': net.bottleneck.parameters(), 'lr': args.lr},
                                 {'params': net.classifier.parameters(), 'lr': args.lr},
                                 {'params': net.base.base.layer3.parameters(), 'lr': 0.2 * args.lr},
                                 {'params': net.base.base.layer4.parameters(), 'lr': 0.25 * args.lr},
                                 {'params': net.base.pe.parameters(), 'lr': 0.1 * args.lr},
                                 {'params': net.transform.parameters(), 'lr': 0.25 * args.lr},],weight_decay=args.weight_decay,nesterov=True,momentum=0.9)
    return optimizer


    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epoch if epoch <= args.warm_up_epoch else \
          0.5 * (math.cos((epoch - args.warm_up_epoch) / (args.max_epoch - args.warm_up_epoch) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


def adjust_learning_rate(epoch,optimizer,args):
    lr = args.lr
    lr_warm_up_eopch = args.warm_up_epoch
    lr_step_epoch = args.lr_step_epoch
    lr_step_ratio = args.lr_step_ratio
    if epoch < lr_warm_up_eopch:
        cur_lr = lr * epoch/lr_warm_up_eopch
    else:
        for i in range(0,len(lr_step_epoch)):
            if epoch < lr_step_epoch[i]:
                break
        cur_lr = lr*lr_step_ratio[i]
        
    optimizer.param_groups[0]['lr'] = cur_lr*0.1
    optimizer.param_groups[1]['lr'] = cur_lr
    optimizer.param_groups[2]['lr'] = cur_lr
    optimizer.param_groups[3]['lr'] = cur_lr * 0.2
    optimizer.param_groups[4]['lr'] = cur_lr * 0.25
    optimizer.param_groups[5]['lr'] = cur_lr * 0.1
    optimizer.param_groups[6]['lr'] = cur_lr * 0.25

    
    return optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],optimizer

def prepare_dataset(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        RandomErasing(probability = args.erasing_p, mean=[0.0, 0.0, 0.0]),
])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize])
    
    # corruption_transform_test = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((args.img_h, args.img_w)),
    #     corruption_transform(),
    #     transforms.ToTensor(),
    #     normalize])
    end = time.time()

    dataset = args.dataset
    if dataset == 'sysu':
        data_path = args.data_path
        log_path = args.log_path + 'sysu_log/'
        test_mode = [1, 2]  # thermal to visible
    elif dataset == 'regdb':
        data_path = args.data_path
        log_path = args.log_path + 'regdb_log/'
        test_mode = [1, 2]  # visible to thermal
    elif dataset == 'llcm':
        data_path = args.data_path
        log_path = args.log_path + 'llcm_log/'
        test_mode = [1, 2]  # visible to thermal
        
    # visible to thermal
    #test_mode = [2, 1]
    # thermal to visible
    # test_mode = [1, 2]
    checkpoint_path = args.model_path
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.isdir(args.vis_log_path):
        os.makedirs(args.vis_log_path)

    if dataset == 'sysu':
        # training set
        trainset = SYSUData(data_path, args=args)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    elif dataset == 'regdb':
        # training set
        trainset = RegDBData(data_path, args.trial)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal') #visible
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible') #thermal
        
    elif dataset == 'llcm':
        # training set
        trainset = LLCMData(data_path, args.trial,transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False,)
    # query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False,)

    n_class = len(np.unique(trainset.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    logging.info('data init success!')

    print('Dataset {} statistics:'.format(dataset))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    settings = [n_class,nquery,ngall,test_mode,dataset]
    label = [query_label,gall_label]
    cam = [query_cam,gall_cam]
    pos = [color_pos,thermal_pos]

    #return settings,label,pos,gall_loader,query_loader,trainset
    return settings,label,cam,pos,gall_loader,query_loader,trainset

    
    
        
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# class GeMP(nn.Module):
#     def __init__(self, power=3):
#         super(GeMP, self).__init__()
#         self.power = power

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.view(b, c, -1)
#         x_pool = (torch.mean(x ** self.power, dim=-1) + 1e-12) ** (1 / self.power)
#         return x_pool

class GeMP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
        super(GeMP, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        p, eps = self.p, self.eps
        if x.ndim != 2:
            batch_size, fdim = x.shape[:2]
            x = x.view(batch_size, fdim, -1)
        return (torch.mean(x ** p, dim=-1) + eps) ** (1 / p)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos


def GenCamIdx(gall_img, gall_label, mode):
    if mode == 'indoor':
        camIdx = [1, 2]
    else:
        camIdx = [1, 2, 4, 5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))

    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k, v in enumerate(gall_label) if v == unique_label[i] and gall_cam[k] == camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos


def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
        # cam_id = 2
        gall_cam.append(cam_id)

    return np.array(gall_cam)


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], int(num_pos))
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], int(num_pos))

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    pass


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class CMAlign(nn.Module):
    def __init__(self, batch_size, num_pos, t=50):
        super(CMAlign, self).__init__()
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.criterion = nn.TripletMarginLoss(margin=0.3, reduce=False, p=2)
        self.t = t

    def _random_pairs(self):
        '''
        根据 batch size 和 num_pos计算positive pair和 negative pair的索引。
        例如batch size = 8 num_pos = 4,则一个ID送进来4张图片(同一模态)，一次进来8个ID，则一共有32张图片
        对于positive pair,保证数组44成组合。
        对于negative pair 随机选。
        '''
        batch_size = self.batch_size
        num_pos = self.num_pos

        pos = []
        for batch_index in range(batch_size):
            pos_idx = random.sample(list(range(num_pos)), num_pos)
            pos_idx = np.array(pos_idx) + num_pos * batch_index
            pos = np.concatenate((pos, pos_idx))
        pos = pos.astype(int)

        neg = []
        for batch_index in range(batch_size):
            batch_list = list(range(batch_size))
            batch_list.remove(batch_index)

            batch_idx = random.sample(batch_list, num_pos)
            neg_idx = random.sample(list(range(num_pos)), num_pos)

            batch_idx, neg_idx = np.array(batch_idx), np.array(neg_idx)
            neg_idx = batch_idx * num_pos + neg_idx
            neg = np.concatenate((neg, neg_idx))
        neg = neg.astype(int)

        return {'pos': pos, 'neg': neg}

    def _define_pairs(self):
        '''
        生成RGB图像、IR图像的正负样本对索引
        '''

        pairs_v = self._random_pairs()
        pos_v, neg_v = pairs_v['pos'], pairs_v['neg']

        pairs_t = self._random_pairs()
        pos_t, neg_t = pairs_t['pos'], pairs_t['neg']

        pos_v += self.batch_size * self.num_pos
        neg_v += self.batch_size * self.num_pos

        return {'pos': np.concatenate((pos_v, pos_t)), 'neg': np.concatenate((neg_v, neg_t))}

    def feature_similarity(self, feat_q, feat_k):
        '''
        使用矩阵乘法，计算特征q与k的匹配度。
        '''
        batch_size, fdim, h, w = feat_q.shape
        # B C L
        feat_q = feat_q.resize(batch_size, fdim, h * w)
        feat_k = feat_k.resize(batch_size, fdim, h * w)
        # torch.bmm:矩阵乘法
        feature_sim = torch.bmm(F.normalize(feat_q, dim=1).permute(0, 2, 1), F.normalize(feat_k, dim=1))
        return feature_sim

    def matching_probability(self, feature_sim):

        M, _ = feature_sim.max(dim=-1, keepdim=True)

        feature_sim = feature_sim - M
        exp = torch.exp(self.t * feature_sim)
        exp_sum = exp.sum(dim=-1, keepdim=True)
        return exp / exp_sum

    def soft_warping(self, matching_pr, feat_k):
        batch_size, fdim, h, w = feat_k.shape
        # B C L
        feat_k = feat_k.resize(batch_size, fdim, h * w)
        # 矩阵乘法
        feat_warp = torch.bmm(matching_pr, feat_k.permute(0, 2, 1))
        # 还原为 B C H W
        feat_warp = feat_warp.permute(0, 2, 1).resize(batch_size, fdim, h, w)

        return feat_warp

    def reconstruct(self, mask, feat_warp, feat_q):

        return mask * feat_warp + (1.0 - mask) * feat_q

    def compute_mask(self, feat):
        batch_size, fdim, h, w = feat.shape
        # 求feat第一维度上的范数，就是按行求，每一行求出一个，作为norms的列
        norms = torch.norm(feat, p=2, dim=1).view(batch_size, h * w)
        # min-max归一化。
        norms -= norms.min(dim=-1, keepdim=True)[0]
        norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
        # 再还原为 B C H W,由于是按照dim = 1求的norm，则通道变为1。
        mask = norms.resize(batch_size, 1, h, w)

        return mask.detach()

    def compute_comask(self, matching_pr, mask_q, mask_k):
        # mask_q >mask_K
        batch_size, mdim, h, w = mask_q.shape
        mask_q = mask_q.resize(batch_size, -1, 1)
        mask_k = mask_k.resize(batch_size, -1, 1)

        comask = mask_q * torch.bmm(matching_pr, mask_k)

        comask = comask.resize(batch_size, -1)
        comask -= comask.min(dim=-1, keepdim=True)[0]
        comask /= comask.max(dim=-1, keepdim=True)[0] + 1e-12
        comask = comask.resize(batch_size, mdim, h, w)

        return comask.detach()

    def forward(self, feat_v, feat_t):
        '''
        Args:
            feat_v: feat[:batch_size//2]
            feat_t: feat[batch_size//2:]

        Returns: {'feat': feat_recon_pos, 'loss': loss}

        '''
        # 把总的feature再合并到一起
        feat = torch.cat([feat_v, feat_t], dim=0)
        # 计算总体特征的mask，即按照通道求二范数之后再归一化。
        mask = self.compute_mask(feat)

        pairs = self._define_pairs()
        # 获取正样本对和负样本对的索引,其中pos_idx为跨模态的同一ID
        # neg_idx 为跨模态的不同id
        pos_idx, neg_idx = pairs['pos'], pairs['neg']

        # 获取实际的positive样本张量。
        feat_target_pos = feat[pos_idx]
        # 计算positive样本和总体特征的匹配度
        feature_sim = self.feature_similarity(feat, feat_target_pos)
        # 计算匹配概率,matching_probability是一个softmax函数，把距离转化为概率。
        matching_pr = self.matching_probability(feature_sim)
        # 将mask[pos_idx]乘以匹配概率，然后再和mask相乘，计算出comask
        comask_pos = self.compute_comask(matching_pr, mask, mask[pos_idx])
        # 使用matching_pr聚合feat_target_pos中的特征。
        feat_warp_pos = self.soft_warping(matching_pr, feat_target_pos)

        # mask*feat_warp_pos + (1.0-mask)*feat
        feat_recon_pos = self.reconstruct(mask, feat_warp_pos, feat)

        # negative
        feat_target_neg = feat[neg_idx]
        feature_sim = self.feature_similarity(feat, feat_target_neg)
        matching_pr = self.matching_probability(feature_sim)

        feat_warp = self.soft_warping(matching_pr, feat_target_neg)
        feat_recon_neg = self.reconstruct(mask, feat_warp, feat)
        # 将loss细分至每一像素。
        loss = torch.mean(comask_pos * self.criterion(feat, feat_recon_pos, feat_recon_neg))

        # experiment:
        # mask_v = self.compute_mask(feat[:b//2])
        # pos_v = pos_idx[:b//2]
        # feat_target_pos_v = feat[pos_v]
        # feature_sim_v = self.feature_similarity(feat_v, feat_target_pos_v)
        # matching_pr_v = self.matching_probability(feature_sim_v)
        # comask_pos_v = self.compute_comask(matching_pr_v, mask_v, mask_v[pos_v])
        #
        # mask_t = self.compute_mask(feat[b//2:])
        # pos_t = pos_idx[b//2:]
        # feat_target_pos = feat[pos_t]
        # feature_sim_t = self.feature_similarity(feat_t, feat_target_pos)
        # matching_pr_t = self.matching_probability(feature_sim_t)
        # comask_pos_t = self.compute_comask(matching_pr_t, mask_t, mask_t[pos_t])
        #
        #
        # loss_t1 = torch.mean(comask_pos_v* self.criterion(feat_v,feat_recon_pos[b//2:],feat_recon_neg[b//2:]))
        #
        # loss_t2 = torch.mean(comask_pos_t * self.criterion(feat_t, feat_recon_pos[:b//2], feat_recon_neg[:b//2]))
        #
        # loss = loss + 0.01 *loss_t1 + 0.01 * loss_t2

        return {'feat': feat_recon_pos, 'loss': loss}


class CM_Attention(nn.Module):
    def __init__(self, batch_size, num_pos, head, t=50):
        super(CM_Attention, self).__init__()
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.criterion = nn.TripletMarginLoss(margin=0.3, reduce=False, p=2)
        self.t = t
        self.head = head

    def _random_pairs(self):

        batch_size = self.batch_size
        num_pos = self.num_pos
        pos = []
        for batch_index in range(batch_size):
            pos_idx = random.sample(list(range(num_pos)), num_pos)
            pos_idx = np.array(pos_idx) + num_pos * batch_index
            pos = np.concatenate((pos, pos_idx))
        pos = pos.astype(int)
        neg = []
        for batch_index in range(batch_size):
            batch_list = list(range(batch_size))
            batch_list.remove(batch_index)
            batch_idx = random.sample(batch_list, num_pos)
            neg_idx = random.sample(list(range(num_pos)), num_pos)
            batch_idx, neg_idx = np.array(batch_idx), np.array(neg_idx)
            neg_idx = batch_idx * num_pos + neg_idx
            neg = np.concatenate((neg, neg_idx))
        neg = neg.astype(int)
        return {'pos': pos, 'neg': neg}

    def _define_pairs(self):

        pairs_v = self._random_pairs()
        pos_v, neg_v = pairs_v['pos'], pairs_v['neg']
        pairs_t = self._random_pairs()
        pos_t, neg_t = pairs_t['pos'], pairs_t['neg']
        pos_t += self.batch_size * self.num_pos
        neg_t += self.batch_size * self.num_pos

        return {'pos': np.concatenate((pos_v, pos_t)), 'neg': np.concatenate((neg_v, neg_t))}

    def K_X_Q(self, feat_q, feat_k):
        batch_size, heads, fdim_perhead, h, w = feat_q.shape
        feat_q = feat_q.view(batch_size, heads, fdim_perhead, -1).transpose(2, 3)
        feat_k = feat_k.view(batch_size, heads, fdim_perhead, -1).transpose(2, 3)
        L2_sim = F.normalize(feat_q, dim=3) @ (F.normalize(feat_k, dim=3).transpose(-2, -1))

        # softmax operation
        M, _ = L2_sim.max(dim=-1, keepdim=True)
        L2_sim = L2_sim - M  # for numerical stability
        exp = torch.exp(self.t * L2_sim)
        exp_sum = exp.sum(dim=-1, keepdim=True)

        return exp / exp_sum

    def compute_warp(self, KQ, feat_k):

        batch_size, num_heads, fdim_per_head, h, w = feat_k.shape
        feat_k = feat_k.view(batch_size, num_heads, fdim_per_head, h * w)
        feat_warp = KQ @ (feat_k.permute(0, 1, 3, 2))
        feat_warp = feat_warp.permute(0, 1, 3, 2).view(batch_size, num_heads, fdim_per_head, h, w)

        return feat_warp

    def reconstruct(self, mask, feat_warp, feat_q):

        return mask * feat_warp + (1.0 - mask) * feat_q

    def compute_mask(self, feat):
        batch_size, num_heads, fdim_perhead, h, w = feat.shape
        # 求feat第一维度上的范数，就是按行求，每一行求出一个，作为norms的列
        norms = torch.norm(feat, p=2, dim=2).view(batch_size, num_heads, h * w)
        # min-max归一化。
        norms -= norms.min(dim=-1, keepdim=True)[0]
        norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
        # 再还原为 B C H W,由于是按照dim = 1求的norm，则通道变为1。
        mask = norms.view(batch_size, num_heads, 1, h, w)
        return mask.detach()

    def compute_comask(self, KQ, mask_q, mask_k):

        # mask = norms.view(batch_size, num_heads, 1, h, w)

        batch_size, num_heads, fdim_perhead, h, w = mask_q.shape
        mask_q = mask_q.view(batch_size, num_heads, fdim_perhead, h * w).transpose(-2, -1)
        mask_k = mask_k.view(batch_size, num_heads, fdim_perhead, h * w).transpose(-2, -1)
        comask = mask_q * (KQ @ mask_k)
        comask = comask.view(batch_size, num_heads, h * w)
        # min_max scale
        comask = torch.norm(comask, p=2, dim=1)
        comask -= comask.min(dim=-1, keepdim=True)[0]
        comask /= comask.max(dim=-1, keepdim=True)[0] + 1e-12
        comask = comask.view(batch_size, fdim_perhead, h, w)

        return comask.detach()

    def forward(self, feat_v, feat_t):

        feat = torch.cat([feat_v, feat_t], dim=0)
        batch_size, fdim, h, w = feat.shape
        pairs = self._define_pairs()
        pos_idx, neg_idx = pairs['pos'], pairs['neg']
        feat_target_pos = feat[pos_idx]

        #  multi-head pre-process
        feat_target_pos = feat_target_pos.reshape(batch_size, self.head, fdim // self.head, h, w)
        feat_mh = feat.reshape(batch_size, self.head, fdim // self.head, h, w)
        mask = self.compute_mask(feat_mh)

        KQ = self.K_X_Q(feat_mh, feat_target_pos)
        comask_pos = self.compute_comask(KQ, mask, mask[pos_idx])
        feat_warp_pos = self.compute_warp(KQ, feat_target_pos)
        feat_recon_pos = self.reconstruct(mask, feat_warp_pos, feat_mh)
        # revierse
        feat_recon_pos = feat_recon_pos.reshape(batch_size, fdim, h, w)

        feat_target_neg = feat[neg_idx]
        feat_target_neg = feat_target_neg.reshape(batch_size, self.head, fdim // self.head, h, w)
        KQ = self.K_X_Q(feat_mh, feat_target_neg)
        feat_warp = self.compute_warp(KQ, feat_target_neg)
        feat_recon_neg = self.reconstruct(mask, feat_warp, feat_mh)
        # reverse
        feat_recon_neg = feat_recon_neg.reshape(batch_size, fdim, h, w)

        # compute loss
        loss = torch.mean(comask_pos * self.criterion(feat, feat_recon_pos, feat_recon_neg))

        return {'feat': feat_recon_pos, 'loss': loss}


class CMAlign_multi_head(nn.Module):
    def __init__(self, batch_size=8, num_pos=4, t=50):
        super(CMAlign_multi_head, self).__init__()
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.criterion = nn.TripletMarginLoss(margin=0.65, reduce=False, p=2)
        self.t = t

    def _random_pairs(self):

        batch_size = self.batch_size
        num_pos = self.num_pos

        pos = []
        for batch_index in range(batch_size):
            pos_idx = random.sample(list(range(num_pos)), num_pos)
            pos_idx = np.array(pos_idx) + num_pos * batch_index
            pos = np.concatenate((pos, pos_idx))
        pos = pos.astype(int)

        neg = []
        for batch_index in range(batch_size):
            batch_list = list(range(batch_size))
            batch_list.remove(batch_index)

            batch_idx = random.sample(batch_list, num_pos)
            neg_idx = random.sample(list(range(num_pos)), num_pos)

            batch_idx, neg_idx = np.array(batch_idx), np.array(neg_idx)
            neg_idx = batch_idx * num_pos + neg_idx
            neg = np.concatenate((neg, neg_idx))
        neg = neg.astype(int)

        return {'pos': pos, 'neg': neg}

    def _define_pairs(self):
        '''
        生成RGB图像、IR图像的正负样本对索引
        '''

        pairs_v = self._random_pairs()
        pos_v, neg_v = pairs_v['pos'], pairs_v['neg']

        pairs_t = self._random_pairs()
        pos_t, neg_t = pairs_t['pos'], pairs_t['neg']

        pos_t += self.batch_size * self.num_pos
        neg_t += self.batch_size * self.num_pos

        return {'pos': np.concatenate((pos_v, pos_t)), 'neg': np.concatenate((neg_v, neg_t))}

    def feature_similarity(self, feat_q, feat_k):

        # feat_q: feat_mh, feat_k: feat_target_pos
        '''
        使用矩阵乘法，计算特征q与k的匹配度。
        '''
        batch_size, heads, fdim_perhead, h, w = feat_q.shape
        # B heads h*w fdim_perhead
        feat_q = feat_q.view(batch_size, heads, fdim_perhead, -1).transpose(2, 3)
        feat_k = feat_k.view(batch_size, heads, fdim_perhead, -1).transpose(2, 3)

        # feature_sim = torch.bmm(F.normalize(feat_q, dim=1).permute(0, 2, 1), F.normalize(feat_k, dim=1))
        feature_sim = F.normalize(feat_q, dim=3) @ (F.normalize(feat_k, dim=3).transpose(-2, -1))
        return feature_sim

    def matching_probability(self, feature_sim):

        M, _ = feature_sim.max(dim=-1, keepdim=True)
        feature_sim = feature_sim - M  # for numerical stability
        exp = torch.exp(self.t * feature_sim)
        exp_sum = exp.sum(dim=-1, keepdim=True)
        return exp / exp_sum

    def soft_warping(self, matching_pr, feat_k):

        batch_size, num_heads, fdim_per_head, h, w = feat_k.shape
        feat_k = feat_k.view(batch_size, num_heads, fdim_per_head, h * w)
        feat_warp = matching_pr @ (feat_k.permute(0, 1, 3, 2))
        feat_warp = feat_warp.permute(0, 1, 3, 2).view(batch_size, num_heads, fdim_per_head, h, w)
        return feat_warp

    def reconstruct(self, mask, feat_warp, feat_q):

        return mask * feat_warp + (1.0 - mask) * feat_q

    def compute_mask(self, feat):
        batch_size, num_heads, fdim_perhead, h, w = feat.shape
        # 求feat第一维度上的范数，就是按行求，每一行求出一个，作为norms的列
        norms = torch.norm(feat, p=2, dim=2).view(batch_size, num_heads, h * w)
        # min-max归一化。
        norms -= norms.min(dim=-1, keepdim=True)[0]
        norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
        # 再还原为 B C H W,由于是按照dim = 1求的norm，则通道变为1。
        mask = norms.view(batch_size, num_heads, 1, h, w)

        return mask.detach()

    def compute_comask(self, matching_pr, mask_q, mask_k):
        # matching_pr: KXQ, mask_q: mask, mask_k: mask[pos_idx]

        batch_size, num_heads, fdim_perhead, h, w = mask_q.shape
        mask_q = mask_q.view(batch_size, num_heads, fdim_perhead, h * w).transpose(-2, -1)
        mask_k = mask_k.view(batch_size, num_heads, fdim_perhead, h * w).transpose(-2, -1)

        comask = mask_q * (matching_pr @ mask_k)

        comask = comask.view(batch_size, num_heads, h * w)
        comask = torch.norm(comask, p=2, dim=1)
        comask -= comask.min(dim=-1, keepdim=True)[0]
        comask /= comask.max(dim=-1, keepdim=True)[0] + 1e-12
        comask = comask.view(batch_size, fdim_perhead, h, w)

        return comask.detach()

    def forward(self, feat_v, feat_t):
        '''
        Args:
            feat_v: feat[:batch_size//2]
            feat_t: feat[batch_size//2:]

        Returns: {'feat': feat_recon_pos, 'loss': loss}

        head = 4
        '''
        head = 4
        # 把总的feature再合并到一起
        feat = torch.cat([feat_v, feat_t], dim=0)
        # 计算总体特征的mask，即按照通道求二范数之后再归一化。
        batch_size, fdim, h, w = feat.shape
        pairs = self._define_pairs()
        # 获取正样本对和负样本对的索引,其中pos_idx为跨模态的同一ID
        # neg_idx 为跨模态的不同id
        pos_idx, neg_idx = pairs['pos'], pairs['neg']

        # 获取实际的positive样本张量。
        feat_target_pos = feat[pos_idx]
        # 计算positive样本和总体特征的匹配度
        feat_target_pos = feat_target_pos.reshape(batch_size, head, fdim // head, h, w)
        feat_mh = feat.reshape(batch_size, head, fdim // head, h, w)

        mask = self.compute_mask(feat_mh)

        feature_sim = self.feature_similarity(feat_mh, feat_target_pos)
        # 计算匹配概率,matching_probability是一个softmax函数，把距离转化为概率。
        matching_pr = self.matching_probability(feature_sim)
        # 将mask[pos_idx]乘以匹配概率，然后再和mask相乘，计算出comask
        comask_pos = self.compute_comask(matching_pr, mask, mask[pos_idx])
        # 使用matching_pr聚合feat_target_pos中的特征。
        feat_warp_pos = self.soft_warping(matching_pr, feat_target_pos)

        # mask*feat_warp_pos + (1.0-mask)*feat
        feat_recon_pos = self.reconstruct(mask, feat_warp_pos, feat_mh)
        feat_recon_pos = feat_recon_pos.reshape(batch_size, fdim, h, w)

        # negative
        feat_target_neg = feat[neg_idx]

        feat_target_neg = feat_target_neg.reshape(batch_size, head, fdim // head, h, w)
        feature_sim = self.feature_similarity(feat_mh, feat_target_neg)
        matching_pr = self.matching_probability(feature_sim)

        feat_warp = self.soft_warping(matching_pr, feat_target_neg)
        feat_recon_neg = self.reconstruct(mask, feat_warp, feat_mh)
        feat_recon_neg = feat_recon_neg.reshape(batch_size, fdim, h, w)
        # 将loss细分至每一像素。
        loss = torch.mean(comask_pos * self.criterion(feat, feat_recon_pos, feat_recon_neg))

        return {'feat': feat_recon_pos, 'loss': loss}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('LayerNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

            
            

