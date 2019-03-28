import os
import random
import torch

import numpy as np
import argparse
from distutils.version import LooseVersion

from torch import nn

from build_model import FCN_GCN

def main(args):
    crit = nn.NLLLoss(ignore_index=-1)

    FCN_GCN_model = FCN_GCN(num_classes=5)

    model_parameters = filter(lambda p: p.requires_grad, FCN_GCN_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters:', params)

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch_encoder', default='resnet50dilated',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/train.odgt')
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')

    # optimization related arguments
    parser.add_argument('--gpus', default='0-3',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=5000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=2e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--deep_sup_scale', default=0.4, type=float,
                        help='the weight of deep supervision loss')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=[300, 375, 450, 525, 600],
                        nargs='+', type=int,
                        help='input image size of short edge (int or list)')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # Parse gpu ids
    args.batch_size =0

    args.max_iters = args.epoch_iters * args.num_epoch
    args.running_lr_encoder = args.lr_encoder
    args.running_lr_decoder = args.lr_decoder

    args.arch_encoder = args.arch_encoder.lower()
    args.arch_decoder = args.arch_decoder.lower()

    # Model ID
    args.id += '-' + args.arch_encoder
    args.id += '-' + args.arch_decoder
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgMaxSize' + str(args.imgMaxSize)
    args.id += '-paddingConst' + str(args.padding_constant)
    args.id += '-segmDownsampleRate' + str(args.segm_downsampling_rate)
    args.id += '-LR_encoder' + str(args.lr_encoder)
    args.id += '-LR_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)
    if args.fix_bn:
        args.id += '-fixBN'
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)