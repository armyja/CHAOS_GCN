# 1. convert train set to trainable files (2d and 3d)
# 2. create train list & test list
# 3.

import argparse
import os
import re
import shutil
import time

import numpy as np

from utils import *


def get_patients(path):
    patients = []
    regex = '^\d+$'
    for x in os.listdir(path):
        if re.match(regex, x):
            patients.append(x)

    return patients


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Parse CHAOS dataset')

    parser.add_argument(
        '--folds',
        default=4,
        type=int,
        help='')
    parser.add_argument(
        '--train_set_dir',
        default='/media/jeffrey/D/CHAOS/Train_Sets/MR',
        help='', )
    parser.add_argument(
        '--root_dir',
        default='/media/jeffrey/D/CHAOS',
        help='', )
    parser.add_argument(
        '--organ_number',
        default=4,
        type=int,
        help='')

    return parser.parse_args()


def training_list(args):

    image_list = []
    image_directory = []
    image_filename =[]
    image_keyword = '.dcm'
    image_path = args.train_set_dir
    for directory, _, file_ in os.walk(image_path):
        for filename in sorted(file_):
            # no label_keyword
            if image_keyword in filename:

                if directory not in image_directory:
                    image_directory.append(directory)

                image_list.append(os.path.join(directory, filename))
                image_filename.append(os.path.splitext(filename)[0])

    label_list = []
    label_directory = []
    label_filename = []
    label_keyword = '.png'
    label_path = args.train_set_dir
    for directory, _, file_ in os.walk(label_path):
        # T1DUAL (in, label) (out, label)
        # T2SPIR (img, label)
        repeat = 1
        if 'T1DUAL' in directory:
            repeat = 2

        for i in range(repeat):
            for filename in sorted(file_):
                # no label_keyword
                if label_keyword in filename:

                    if directory not in label_directory:
                        for j in range(repeat):
                            label_directory.append(directory)

                    label_list.append(os.path.join(directory, filename))
                    label_filename.append(os.path.splitext(filename)[0])

    if len(image_list) != len(label_list):
        exit(f'{len(image_list)} {len(label_list)}Error: the number of labels and the number of images are not equal!')

    if len(image_directory) != len(label_directory):
        exit(f'{len(image_directory)} {len(label_directory)}Error: the number of labels and the number of images are not equal!')

    total_samples = len(image_directory)

    # volume_id slice_id patient_id image_path label_path
    list_path = list_path_from_root(args.root_dir)
    try:
        shutil.rmtree(list_path)
    except OSError as e:
        print(e)
    os.makedirs(list_path, exist_ok=True)
    print('    Writing training lists: ' )

    for i in range(total_samples):
        print('Processing ' + str(i + 1) + ' out of ' + str(total_samples) + ' files.')
        m_image_list = list(filter(lambda x: image_directory[i] in x, image_list))
        m_label_list = list(filter(lambda x: label_directory[i] in x, label_list))
        if 'T1DUAL' in label_directory[i]:
            m_label_list = m_label_list[:int(len(m_label_list)/2)]
        assert (len(m_image_list) == len(m_label_list))

        slice_number = len(m_image_list)
        m_average = np.zeros(slice_number, dtype=np.float)
        m_sum = np.zeros((slice_number, args.organ_number + 1), dtype=np.int)
        for j in range(slice_number):
            m_image_filename = m_image_list[j]
            m_label_filename = m_label_list[j]
            m_image = dcm2npy(m_image_filename)
            m_label = png2npy(m_label_filename)
            m_average[j] = float(m_image.sum()) / (m_image.shape[0] * m_image.shape[1])

            for o in range(1, args.organ_number + 1):
                m_sum[j, o] = (is_organ(m_label, o)).sum()

        output = open(list_training_all(list_path), 'a+')

        for j in range(0, slice_number):
            output.write(str(i) + ' ' + str(j))
            output.write(' ' + m_image_list[j] + ' ' + m_label_list[j])
            output.write(' ' + str(m_average[j]))
            for o in range(1, args.organ_number + 1):
                output.write(' ' + str(m_sum[j, o]))
            output.write('\n')

        output.close()
        folds = args.folds

    print('Writing training image list.')
    for f in range(folds):
        list_training_ = training_set_filename(list_path, f)
        output = open(list_training_, 'w')
        for i in range(total_samples):
            if in_training_set(total_samples, i, folds, f):
                output.write(str(i) + ' ' + image_directory[i] + ' ' + label_directory[i] + '\n')
        output.close()
    print('Writing testing image list.')
    for f in range(folds):
        list_testing_ = testing_set_filename(list_path, f)
        output = open(list_testing_, 'w')
        for i in range(total_samples):
            if not in_training_set(total_samples, i, folds, f):
                output.write(str(i) + ' ' + image_directory[i] + ' ' + label_directory[i] + '\n')
        output.close()
    print('Initialization is done.')
    exit(0)

def main():
    args = parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    training_list(args)
    pass


if __name__ == '__main__':
    main()
