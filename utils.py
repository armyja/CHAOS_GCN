import PIL.Image
import os

import SimpleITK as sitk
import numpy as np


####################################################################################################
# returning the binary label map by the organ ID (especially useful under overlapping cases)
#   label: the label matrix
#   organ_ID: the organ ID
def is_organ(label, organ_ID):
    return label == organ_ID


####################################################################################################
# determining if a sample belongs to the training set by the fold number
#   total_samples: the total number of samples
#   i: sample ID, an integer in [0, total_samples - 1]
#   folds: the total number of folds
#   current_fold: the current fold ID, an integer in [0, folds - 1]
def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (start_index <= i < end_index)


####################################################################################################
# returning the filename of the training set according to the current fold ID
def training_set_filename(list_path, current_fold):
    return os.path.join(list_path, 'training_' + 'FD' + str(current_fold) + '.txt')


####################################################################################################
# returning the filename of the testing set according to the current fold ID
def testing_set_filename(list_path, current_fold):
    return os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')


def snapshot_path_from_root(root):
    return os.path.join(root, 'snapshots')


def list_path_from_root(root):
    return os.path.join(root, 'lists')


def list_training_all(list_path):
    return os.path.join(list_path, 'training.txt')


def dcm2npy(dcm):
    image = sitk.ReadImage(dcm)
    # (35 256 256) to (256 256 35)
    array = sitk.GetArrayFromImage(image).transpose(1, 2, 0)
    data = np.zeros(array.shape, dtype=np.int16)
    data[:, :, :] = array
    return data

def npy2npy(image, mask=False):
    array = np.load(image)
    array = array.reshape(array.shape[0],array.shape[1], 1)
    data = np.zeros(array.shape, dtype=np.int16)
    data[:, :, :] = array
    if mask is True:
        unique_values_mask = np.unique(data)
        gt_mask = np.zeros_like(data).astype(np.int64)
        for unique_value in unique_values_mask:
            gt_mask[data == unique_value] = class_mapping(unique_value)
        return gt_mask

    return data


# Mapping
# 0 is background
# 1 is liver
# 2 is right kidney
# 3 is left kidney
# 4 is spleen


def class_mapping(input_value):
    if 55 < input_value <= 70:
        return 1
    elif 110 < input_value <= 135:
        return 2
    elif 175 < input_value <= 200:
        return 3
    elif 240 < input_value <= 255:
        return 4
    else:
        return 0


def png2npy(png):
    gt_arr = np.asarray(PIL.Image.open(png))
    unique_values_mask = np.unique(gt_arr)
    gt_mask = np.zeros_like(gt_arr).astype(np.int64)
    for unique_value in unique_values_mask:
        gt_mask[gt_arr == unique_value] = class_mapping(unique_value)
    return gt_mask
