import argparse
import time
import torch
from datetime import datetime
from glob import glob

import numpy as np
import visdom

import build_model
import data_loader

from utils import *


####################################################################################################
# computing the DSC together with other values based on the label and prediction volumes
# def DSC_computation(label, pred):
# 	pred_sum = pred.sum()
# 	label_sum = label.sum()
# 	inter_sum = np.logical_and(pred, label).sum()
# 	return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum
def DSC_computation(label, pred):
    P = np.zeros(3, dtype=np.uint32)
    ff.DSC_computation(label, pred, P)
    return 2 * float(P[2]) / (P[0] + P[1]), P[2], P[1], P[0]


def parse_args():
    parser = argparse.ArgumentParser()
    # Model related arguments
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
        default=0,
        type=int,
        help='')
    parser.add_argument(
        '--loss_type',
        default='cross_entropy2d',
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

    return parser.parse_args()


def main():
    args = parse_args()
    #######################
    #  USE GPU FOR MODEL  #
    #######################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #
    # snapshot_path = snapshot_path_from_root(args.root_dir)
    # snapshots = os.listdir(snapshot_path)
    #
    # DSC = np.zeros((len(snapshots), len(volume_list)))
    # label = is_organ(label, organ_ID).astype(np.uint8)
    # DSC[t, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_temp)
    test_dataset = data_loader.OrganVolTest(current_fold=args.current_fold,
                                            list_path=list_path_from_root(args.root_dir),
                                            transforms=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    snapshot_files = [
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 20:17:00.783225_937.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 20:17:00.783225_1874.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 20:17:00.783225_2811.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 20:17:00.783225_3748.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 21:11:03.548498_937.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 21:11:03.548498_1874.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 21:11:03.548498_2811.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 21:11:=03.548498_3748.pkl',  # best 8786
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_2019-03-29 21:59:55.931487_937.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190329_223721_937.pkl',  # 7665
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190329_223721_1874.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190329_223721_2811.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190329_223721_3748.pkl',  # 8666
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190329_223721_4685.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190329_223721_5622.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190329_223721_6559.pkl'
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190330_154159_1417.pkl'  # 7663
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190330_124553_30001.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190330_124553_47001.pkl',
        # '/media/jeffrey/D/CHAOS/snapshots/main_GCN_20190330_124553_17001.pkl',  # 8887
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_20190330_124553_17001.pkl',  # 8887
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190401_072416_1001.pkl',  # 8786
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190401_075415_2001.pkl',  # 8786
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190401_082623_2001.pkl',  # 8775 _1
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190401_092224_2001.pkl',  # 8786 _2
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190401_104845_2001.pkl',  # 8786 _3
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190401_134340_2001.pkl',  # 8786 _4
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190402_083113_2001.pkl',  # 8786 _5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190402_084907_2001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190402_091848_2001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190402_141629_2001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190402_143132_26001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190403_050144_14001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190403_103014_8001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190403_113937_2001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190403_124518_20001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_All_20190407_090219_17000.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190403_103014_12001.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190421_011307_16000.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190425_084420FCN_GCN_GCN_NO_SE_16000.pkl',  # 8786 _5.5
        '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190521_084932FCN_GCN_ALL_20000.pkl',  # 8786 _5.5
        # '/home/ubuntu/MyFiles/GCN/snapshots/main_GCN_All_20190331_054658_35001.pkl'
    ]
    # snapshot_file =
    for snapshot_file in snapshot_files:
        file_name = snapshot_file.split('/')[-1]

        # net = build_model.FCN_GCN(num_classes=args.organ_number + 1)
        # net.to(device)
        # if 'All' in snapshot_file:
        #     checkpoint = torch.load(snapshot_file)
        #     net.load_state_dict(checkpoint['net'])
        #     net.eval()
        # else:
        #     net.load_state_dict(torch.load(snapshot_file))
        #     net.eval()
        patient = 2 - 1
        # patient = None
        slice_index = 21 - 1
        # slice_index = None
        # test_volume(net, test_loader, test_dataset, file_name, args, patient, slice_index)
        list_path = '/home/ubuntu/MyFiles/GCN/lists'
        current_fold = 1
        image_list = open(testing_set_filename(list_path, current_fold), 'r').read().splitlines()
        testing_image_set = np.zeros((len(image_list)), dtype=np.int)
        testing_image_path = {}
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            testing_image_set[i] = int(s[0])
            p = s[1]
            p = p.replace('Train_Sets_All', 'CHAOS_result_submit_example/Armyja/Task5')
            p = p.split('DICOM')[0] + 'Results'
            testing_image_path[int(s[0])] = p
            print(p)

        # load test result npy array
        test_result_path = '/home/ubuntu/MyFiles/GCN/test'

        # glob
        test_npy = glob(os.path.join(test_result_path, '*.npy'))
        test_npy = sorted(test_npy, key=lambda x: int(os.path.basename(x).split('.npy')[0]))

        for idx, path in enumerate(test_npy):
            # skip T1 in phase
            if idx % 3 == 0:
                pass
            id = int(os.path.basename(path).split('.npy')[0])
            testing_write_path = testing_image_path[id]
            os.makedirs(testing_write_path, exist_ok=True)

            array = np.load(path)
            array = array[0]
            slices = array.shape[0]
            for i in range(slices):
                slice = array[i]
                h, w = slice.shape
                # arr = np.repeat(slice.reshape(1, h, w), 3, axis=0)
                img = PIL.Image.fromarray(slice)
                write_path = os.path.join(testing_write_path, 'img{}.png'.format(i))
                img.save(write_path)

            print(idx, path)

        # read array, save to png

        exit(0)
        test_volume(net, test_loader, test_dataset, file_name, args)


def test_volume(net, test_loader, test_dataset, snapshot_file_name, args, patient=None, slice_index=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Test Code
    vis = visdom.Visdom()
    vis.delete_env(env=f'GCN_test{snapshot_file_name}')
    viz = visdom.Visdom(env=f'GCN_test_{snapshot_file_name}')
    j = 0
    DSC = np.zeros((len(test_loader), args.organ_number), dtype=np.float)
    for images, labels, test_index, width in test_loader:
        if test_index <= 110:
            pass
        # 1 x slices x 3 x h x w np.float32
        n, s, c, h, w = images.size()
        print(test_index, width, h)
        W = 384
        Y = int((width - W) / 2)
        preds = np.zeros((n, s, width, width), dtype=np.uint8)
        labels = labels.numpy().astype(np.uint8)

        if patient is not None:
            if j != patient:
                j += 1
                continue
        for i in range(s):
            img = images[0][i].unsqueeze(0).requires_grad_().to(device)
            if j == patient and slice_index == i:
                outputs = net(img, debug=True, viz=viz, patient=patient, slice_index=slice_index)
            else:
                outputs = net(img)
            predicted = data_loader.decode_output_to_label(outputs)
            predicted = predicted.view(h, w).numpy().astype(np.uint8)
            preds[0][i][Y:Y + W, Y:Y + W] = predicted

        preds[preds == 1] = 63
        preds[preds == 2] = 126
        preds[preds == 3] = 189
        preds[preds == 4] = 252

        np.save('/home/ubuntu/MyFiles/GCN/test/{}.npy'.format(test_index[0].item()), preds)
        # for k in range(args.organ_number):
        #     n_class = k + 1
        #     pred = is_organ(preds, n_class).astype(np.uint8)
        #     label = is_organ(labels, n_class).astype(np.uint8)
        #     inter_sum = (label * pred).sum()
        #     label_sum = label.sum()
        #     pred_sum = pred.sum()
        #     ret = (2 * inter_sum) / (label_sum + pred_sum)
        #     DSC[j][k] = ret
        #     viz.text(f'patient: {j + 1}, n_class: {n_class}:\n' + \
        #              '    DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + \
        #              ' + ' + str(label_sum) + ') = ' + str(ret) + ' .')
        #     print(f'patient: {j + 1}, n_class: {n_class}:\n' + \
        #           '    DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + \
        #           ' + ' + str(label_sum) + ') = ' + str(ret) + ' .')
        j = j + 1

        # 1 xslices x h x w np.int64
        labels = test_dataset.decode_segmap(torch.from_numpy(labels), )
        preds = test_dataset.decode_segmap(torch.from_numpy(preds), )
        all = torch.IntTensor(s * 3, 3, width, width)
        for i in range(s):
            all[i * 3] = torch.from_numpy(preds[i])
            all[i * 3 + 1, :, :, :] = torch.from_numpy(labels[i])
            all[i * 3 + 2, :, Y:Y + W, Y:Y + W] = images[0][i] / 4
        if slice_index is not None:
            viz.images(all[slice_index * 3:slice_index * 3 + 3, :, :, :], 3, 1,
                       opts=dict(title=f'{patient + 1}_{slice_index + 1}'))
        viz.images(all, 6, 1,
                   opts=dict(title=f'{j}_1:{DSC[j - 1][0]}_2:{DSC[j - 1][1]}_3:{DSC[j - 1][2]}_4:{DSC[j - 1][3]}_'))

    # for k in range(args.organ_number):
    #     # viz.text(f'Snapshot {snapshot_file_name}: n_class={k + 1}: average DSC = ' + str(np.mean(DSC[:, k])) + ' .')
    #     print(f'Snapshot {snapshot_file_name}: n_class={k + 1}: average DSC = ' + str(np.mean(DSC[:, k])) + ' .')

    return DSC


if __name__ == '__main__':
    main()
