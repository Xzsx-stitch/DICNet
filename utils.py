import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from loss import OriTripletLoss
from torch.nn import init
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


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

    return settings,label,cam,pos,gall_loader,query_loader,trainset


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


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


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
