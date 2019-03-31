import os
import random
import time
from datetime import datetime

import torch

import numpy as np
import argparse
from distutils.version import LooseVersion

import visdom
from torch import nn

from RSTN_test import test_volume
from loss import loss
from utils import *
import data_loader
import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help="a name for identifying the model")
    parser.add_argument('--epoch',
                        default=8,
                        type=int,
                        help="a name for identifying the model")
    parser.add_argument('--current_fold',
                        default=0,
                        type=int,
                        help="a name for identifying the model")
    parser.add_argument(
        '--root_dir',
        default='/media/jeffrey/D/CHAOS',
        help='', )
    parser.add_argument(
        '--organ_number',
        default=4,
        type=int,
        help='')
    parser.add_argument(
        '--organ_id',
        default=1,
        type=int,
        help='')
    parser.add_argument(
        '--loss_type',
        default='cross_entropy2d',
        help='', )
    parser.add_argument(
        '--resume',
        default='',
        help='', )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='', )
    parser.add_argument(
        '--timestamp',
        default=datetime.fromtimestamp(time.time()),
        help='', )

    parser.add_argument(
        '--optimizer',
        default='Adagrad',
        help='', )

    return parser.parse_args()


def main():
    vis = visdom.Visdom()
    args = parse_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # Todo Step 1: Load Dataset
    '''
    STEP 1: LOADING DATASET
    '''

    train_dataset = data_loader.OrganSeg(current_fold=args.current_fold,
                                         list_path=list_path_from_root(args.root_dir),
                                         n_class=args.organ_number + 1,
                                         organ_id=args.organ_id,
                                         # slice_threshold=0 means training all images and neglect labels
                                         slice_threshold=0,
                                         transforms=True)

    test_dataset = data_loader.OrganVolTest(current_fold=args.current_fold,
                                            list_path=list_path_from_root(args.root_dir),
                                            transforms=None)
    # Todo Step 2: Make Dataset Iterable
    '''
    STEP 2: MAKING DATASET ITERABLE
    '''

    batch_size = args.batch_size
    num_epochs = args.epoch
    n_iters = int(num_epochs * (len(train_dataset) / batch_size))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    # Todo Step 3: Create Model Class
    # Todo Step 4: Instantiate Model Class
    '''
    STEP 4: INSTANTIATE MODEL CLASS
    '''
    # input_dim = 256 * 256 or 288 * 288
    # output_dim = N x C x H x W
    # C = num_classes

    model = build_model.FCN_GCN(num_classes=args.organ_number + 1)

    #######################
    #  USE GPU FOR MODEL  #
    #######################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    iter = 0
    if args.resume != '0':
        # main_GCN_20190330_124553_16001.pkl
        snapshot_resume = args.resume
        iter = int(snapshot_resume.split('_')[-1].split('.')[0])
        args.timestamp = snapshot_resume.split('GCN_')[1].split(f'_{iter}')[0]
        snapshot_path = snapshot_path_from_root(args.root_dir)
        snapshot_file = os.path.join(snapshot_path, snapshot_resume)
        model.load_state_dict(torch.load(snapshot_file))
        model.eval()
        print(f'{snapshot_file} loaded, iter={iter}')

        # Todo Step 5: Instantiate Loss Class

    '''
    STEP 5: INSTANTIATE LOSS CLASS
    '''
    loss_dict = {
        'cross_entropy2d': loss.cross_entropy2d,
        'bootstrapped_cross_entropy2d': loss.bootstrapped_cross_entropy2d,
        'multi_scale_cross_entropy2d': loss.multi_scale_cross_entropy2d,
    }
    criterion = loss_dict[args.loss_type]
    criterion_weights = torch.FloatTensor([1.0, 4.0, 8.0, 8.0, 4.0]).to(device)
    print('criterion_weights', criterion_weights)

    # Todo Step 6: Instantiate Optimizer Class

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate,
                                    )

    # Todo Step 7: Train Model

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters:', params)

    print("Latest Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    flag = 0
    iter_per_epoch = 0
    for epoch in range(num_epochs):
        t = time.time()
        total_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            m_t = time.time()
            #######################
            #  USE GPU FOR MODEL  #
            #######################
            images = images.requires_grad_().to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            m_loss = criterion(outputs, labels, weight=criterion_weights)
            total_loss += m_loss.item()

            # Getting gradients w.r.t. parameters
            m_loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1
            flag += 1
            print('Iteration: {}. Loss: {}, {} seconds elapsed'.format(iter, m_loss.item(), str(time.time() - m_t)))
            del images, labels, outputs, m_loss

            # if iter % 1000 == 0:
            #     # Calculate Accuracy
            #     correct = 0
            #     total = 0
            #     # Iterate through test dataset
            #     _iter = 0
            #     vis.delete_env(env=f'main_GCN_{args.timestamp}_{iter}')
            #     viz = visdom.Visdom(env=f'main_GCN_{args.timestamp}_{iter}')
            #     for images, labels in test_loader:
            #         _iter += 1
            #         #######################
            #         #  USE GPU FOR MODEL  #
            #         #######################
            #         images = images.requires_grad_().to(device)
            #         labels = labels.to(device)
            #         # Forward pass only to get logits/output
            #         outputs = model(images)
            #         # Total number of labels
            #         total += labels.size(0)
            #         #######################
            #         #  USE GPU FOR MODEL  #
            #         #######################
            #         # Total correct predictions
            #         _m_loss = criterion(outputs, labels).item()
            #         correct += _m_loss
            #         print('Test Iteration: {}. Loss: {}'.format(_iter, _m_loss))
            #
            #         predicted = data_loader.decode_output_to_label(outputs)
            #
            #         predicted_img = train_dataset.decode_segmap(predicted)
            #         label_img = train_dataset.decode_segmap(labels.cpu())
            #         viz.image(predicted_img)
            #         viz.image(label_img)
            #         del images, labels, _m_loss, predicted, predicted_img, label_img
            #     avg = correct / total
            #
            #     # Print Loss
            #     print('Iteration: {}. Avg_Loss: {}'.format(iter, avg))
            #     viz.text('Iteration: {}. Avg_Loss: {}'.format(iter, avg))
            if iter % 1000 == 1:
                snapshot_path = snapshot_path_from_root(args.root_dir)
                snapshot_name = f'main_GCN_{args.timestamp}_{iter}.pkl'
                os.makedirs(snapshot_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(snapshot_path, snapshot_name))
                DSC = test_volume(net=model, test_loader=test_loader, test_dataset=test_dataset,
                                  snapshot_file_name=snapshot_name, args=args)

                means = np.zeros(DSC.shape[1])
                for i in range(DSC.shape[1]):
                    means[i] = np.mean(DSC[:, i])
                rank = means.argsort()
                rank[rank == 3] = 2
                rank[rank == 0] = 1
                ratio = 4
                criterion_weights = torch.FloatTensor([1.0, 1.0 * rank[3] * ratio,
                                                       1.0 * rank[2] * ratio,
                                                       1.0 * rank[1] * ratio,
                                                       1.0 * rank[0] * ratio]).to(device)
                print('criterion_weights', criterion_weights)
        if iter_per_epoch == 0:
            iter_per_epoch = flag
        print('Epoch: {}, Iteration: {}. Avg_Loss: {}, {} seconds elapsed'.format(epoch, iter,
                                                                                  total_loss / iter_per_epoch,
                                                                                  str(time.time() - t)))


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'
    main()
