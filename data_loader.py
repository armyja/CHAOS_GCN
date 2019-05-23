import math
import random

import torch

import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import os

from torchvision import transforms

from utils import *


# 1 x n_class x height x width tensor
def decode_output_to_label(temp):
    n, c, h, w = temp.size()
    temp = temp.transpose(1, 2).transpose(2, 3).squeeze(0).view(h, w, c)
    if torch.cuda.is_available():
        temp = temp.cpu()
    temp = temp.argmax(-1)
    temp = torch.LongTensor(temp.view(1, 1, h, w))
    return temp

    # heightxwidth


class OrganSeg(Dataset):
    def __init__(self, current_fold, list_path, n_class, organ_id, slice_threshold=0, transforms=True):
        self.organ_ID = int(organ_id)
        self.n_class = int(n_class)
        self.transforms = transforms
        self.augmentations = None
        image_list = open(training_set_filename(list_path, current_fold), 'r').read().splitlines()

        self.training_image_set = np.zeros((len(image_list)), dtype=np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.training_image_set[i] = int(s[0])

        slice_list = open(list_training_all(list_path), 'r').read().splitlines()
        self.slices = len(slice_list)
        self.image_ID = np.zeros(self.slices, dtype=np.int)
        self.slice_ID = np.zeros(self.slices, dtype=np.int)
        self.image_filename = ['' for l in range(self.slices)]
        self.label_filename = ['' for l in range(self.slices)]
        self.average = np.zeros(self.slices)
        self.pixels = np.zeros(self.slices, dtype=np.int)

        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]
            self.slice_ID[l] = s[1]
            self.image_filename[l] = s[2]  # important
            self.label_filename[l] = s[3]  # important
            self.average[l] = float(s[4])  # pixel value avg
            self.pixels[l] = int(s[organ_id + 5 - 1])  # sum of label
        if 0 < slice_threshold < 1:  # 0.98
            pixels_index = sorted(range(self.slices), key=lambda l: self.pixels[l])
            last_index = int(math.floor((self.pixels > 0).sum() * slice_threshold))
            min_pixels = self.pixels[pixels_index[-last_index]]
        else:  # or set up directly
            min_pixels = slice_threshold
        # slice_threshold = min_pixels = 0 means all organ
        self.active_index = [l for l, p in enumerate(self.pixels)
                             if p >= min_pixels and self.image_ID[l] in self.training_image_set]  # true active
        colors = [  #
            [0, 0, 0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
        ]

        self.label_colours = dict(zip(range(self.n_class), colors))

    def __getitem__(self, index):
        # stuff
        self.index1 = self.active_index[index]
        if '.dcm' in self.image_filename[self.index1]:
            image1 = dcm2npy(self.image_filename[self.index1]).astype(np.float32)
        elif '.npy' in self.image_filename[self.index1]:
            image1 = npy2npy(self.image_filename[self.index1]).astype(np.float32)

        if 'T1DUAL' in self.image_filename[self.index1]:
            self.low_range = 0.0
            self.high_range = 1200.0
        elif 'T2SPIR' in self.image_filename[self.index1]:
            self.low_range = 0.0
            self.high_range = 1800.0

        # set range
        np.minimum(np.maximum(image1, self.low_range, image1), self.high_range, image1)
        if random.randint(0, 1) == 1:
            image1 = self.high_range + self.low_range - image1
        # image1 -= self.low_range
        # image1 /= (self.high_range - self.low_range)

        if '.png' in self.label_filename[self.index1]:
            label1 = png2npy(self.label_filename[self.index1])
        elif '.npy' in self.label_filename[self.index1]:
            label1 = npy2npy(self.label_filename[self.index1], mask=True)

        width = label1.shape[0]
        height = label1.shape[1]

        lbl = label1.reshape(1, width, height)
        img = image1.reshape(1, width, height)

        if self.transforms is not None:
            img, lbl = self.transform(img, lbl)


        width, height = 256, 256
        lbl = lbl.reshape(width, height)
        img = img.reshape(width, height)
        # set rotate
        # rotate_time = random.randint(0, 3)
        # lbl = np.rot90(lbl, rotate_time)
        # img = np.rot90(img, rotate_time)
        # set flip
        # flip_time = random.randint(0, 1)
        # if flip_time == 1:
        #     lbl = lbl.T
        #     img = img.T

        # mix_rate = random.randint(0, 5)
        # if mix_rate >= 8:
        #     length = len(self.active_index)
        #     self.random_index = (self.index1 + random.randint(0, length - 1)) % length
        #     if '.dcm' in self.image_filename[self.random_index]:
        #         image1 = dcm2npy(self.image_filename[self.random_index]).astype(np.float32)
        #     elif '.npy' in self.image_filename[self.random_index]:
        #         image1 = npy2npy(self.image_filename[self.random_index]).astype(np.float32)
        #     np.minimum(np.maximum(image1, self.low_range, image1), self.high_range, image1)
        #
        #     width = image1.shape[0]
        #     height = image1.shape[1]
        #     image1 = image1.reshape(1, width, height)
        #     image1, image1 = self.transform(image1, image1)
        #
        #     width, height = 256, 256
        #     image1 = image1.reshape(width, height)
        #     img = img * 0.6 + image1 * 0.4

        img = np.repeat(img.reshape(1, width, height), 3, axis=0)
        lbl = lbl.reshape(1, width, height)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        img = np.ascontiguousarray(img, dtype=np.float32)
        lbl = np.ascontiguousarray(lbl, dtype=np.int64)
        return img, lbl

    def transform(self, img, lbl):
        W = 256
        H = 256
        if lbl.shape[1] > H and lbl.shape[2] > W:
            X = int((lbl.shape[1] - H) / 2)
            Y = int((lbl.shape[2] - W) / 2)
            lbl = lbl[:, X:X + H, Y:Y + W]
        if img.shape[1] > H and img.shape[2] > W:
            X = int((img.shape[1] - H) / 2)
            Y = int((img.shape[2] - W) / 2)
            img = img[:, X:X + H, Y:Y + W]

        # transformations_train = transforms.Compose([transforms.RandomRotation(10),
        #                                             transforms.RandomHorizontalFlip(),
        #                                             transforms.ToTensor()])
        # img = transformations_train(img)
        # lbl = transformations_train(lbl)
        return img, lbl

    def decode_segmap(self, temp, bias=0):
        n, c, h, w = temp.size()
        temp = temp.view(h, w)
        temp = temp.numpy()
        temp = temp.astype(np.int8)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_class):
            r[temp == l] = self.label_colours[l][0 + bias * 3]
            g[temp == l] = self.label_colours[l][1 + bias * 3]
            b[temp == l] = self.label_colours[l][2 + bias * 3]

        rgb = np.zeros((3, temp.shape[0], temp.shape[1]))
        rgb[0, :, :] = r
        rgb[1, :, :] = g
        rgb[2, :, :] = b
        return rgb

    def __len__(self):
        return len(self.active_index)  # of how many data(images?) you have


class OrganTest(Dataset):
    def __init__(self, current_fold, list_path, transforms=None):
        self.augmentations = None
        self.transforms = transforms
        image_list = open(testing_set_filename(list_path, current_fold), 'r').read().splitlines()

        self.testing_image_set = np.zeros((len(image_list)), dtype=np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.testing_image_set[i] = int(s[0])

        slice_list = open(list_training_all(list_path), 'r').read().splitlines()
        self.slices = len(slice_list)
        self.image_ID = np.zeros(self.slices, dtype=np.int)
        self.pixels = np.zeros(self.slices, dtype=np.int)
        self.image_filename = ['' for l in range(self.slices)]
        self.label_filename = ['' for l in range(self.slices)]
        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]
            self.image_filename[l] = s[2]  # important
            self.label_filename[l] = s[3]  # important
        self.active_index = [l for l, p in enumerate(self.pixels)
                             if self.image_ID[l] in self.testing_image_set]  # true active

    def __getitem__(self, index):
        # stuff
        self.index1 = self.active_index[index]
        image1 = dcm2npy(self.image_filename[self.index1]).astype(np.float32)

        if 'T1DUAL' in self.image_filename[self.index1]:
            self.low_range = 0.0
            self.high_range = 1200.0
        elif 'T2SPIR' in self.image_filename[self.index1]:
            self.low_range = 0.0
            self.high_range = 1800.0

        np.minimum(np.maximum(image1, self.low_range, image1), self.high_range, image1)
        # image1 -= self.low_range
        # image1 /= (self.high_range - self.low_range)

        label1 = png2npy(self.label_filename[self.index1])
        width = label1.shape[0]
        height = label1.shape[1]
        img = np.repeat(image1.reshape(1, width, height), 3, axis=0)
        lbl = label1.reshape(1, width, height)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.transforms is not None:
            img = self.transforms(img)
            lbl = self.transforms(lbl)

        return img, lbl

    def __len__(self):
        return len(self.active_index)


class OrganVolTest(Dataset):
    def __init__(self, current_fold, list_path, transforms=None):
        self.augmentations = None
        self.n_class = 5
        self.transforms = transforms
        image_list = open(testing_set_filename(list_path, current_fold), 'r').read().splitlines()

        self.testing_image_set = np.zeros((len(image_list)), dtype=np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.testing_image_set[i] = int(s[0])

        slice_list = open(list_training_all(list_path), 'r').read().splitlines()
        self.slices = len(slice_list)
        self.image_ID = np.zeros(self.slices, dtype=np.int)
        self.pixels = np.zeros(self.slices, dtype=np.int)
        self.image_filename = ['' for l in range(self.slices)]
        self.label_filename = ['' for l in range(self.slices)]
        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]
            self.image_filename[l] = s[2]  # important
            self.label_filename[l] = s[3]  # important

        colors = [  #
            [0, 0, 0],
            [63, 63, 63],
            [126, 126, 126],
            [189, 189, 189],
            [252, 252, 252],
            [128, 64, 128],
            # [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            # [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [244, 35, 32],
            [152, 251, 52],
            [0, 130, 80],
            [244, 35, 232],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]

        self.label_colours = colors

    def __getitem__(self, index):
        # stuff
        self.index1 = self.testing_image_set[index]
        self.active_index = [l for l, p in enumerate(self.pixels)
                             if self.image_ID[l] == self.index1]  # true active
        if '.dcm' in self.image_filename[self.active_index[0]]:
            tmp = dcm2npy(self.image_filename[self.active_index[0]]).astype(np.float32)
        elif '.npy' in self.image_filename[self.active_index[0]]:
            tmp = npy2npy(self.image_filename[self.active_index[0]]).astype(np.float32)
        # tmp = dcm2npy(self.image_filename[self.active_index[0]]).astype(np.float32)
        width = tmp.shape[0]
        height = tmp.shape[1]
        print(width, height)
        W = 384
        H = 384
        img_vol = np.zeros((len(self.active_index), 3, H, W), dtype=np.float32)
        lbl_vol = np.zeros((len(self.active_index), height, width), dtype=np.int64)
        for idx, id in enumerate(self.active_index):

            if '.dcm' in self.image_filename[id]:
                image1 = dcm2npy(self.image_filename[id]).astype(np.float32)
            elif '.npy' in self.image_filename[id]:
                image1 = npy2npy(self.image_filename[id]).astype(np.float32)

            # image1 = dcm2npy(self.image_filename[id]).astype(np.float32)

            if '.png' in self.label_filename[id]:
                label1 = png2npy(self.label_filename[id])
            elif '.npy' in self.label_filename[id]:
                label1 = npy2npy(self.label_filename[id], mask=True)

            # label1 = png2npy(self.label_filename[id])
            img = np.repeat(image1.reshape(1, width, height), 3, axis=0)
            # lbl = label1.reshape(1, width, height)
            lbl = img[0]
            W = 384
            H = 384

            if height > H and width > W:
                X = int((height - H) / 2)
                Y = int((width - W) / 2)
                img = img[:, X:X + H, Y:Y + W]
            img_vol[idx, :] = img
            lbl_vol[idx, :] = lbl

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.transforms is not None:
            img = self.transforms(img)
            lbl = self.transforms(lbl)

        return img_vol, lbl_vol, self.index1, width

    def __len__(self):
        return len(self.testing_image_set)

    def decode_segmap(self, temp, bias=0):
        n, c, h, w = temp.size()
        temp = temp.view(c, h, w)
        temp = temp.numpy()
        temp = temp.astype(np.uint8)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_class):
            r[temp == l] = self.label_colours[l + bias * self.n_class][0]
            g[temp == l] = self.label_colours[l + bias * self.n_class][1]
            b[temp == l] = self.label_colours[l + bias * self.n_class][2]

        l = 0
        r[temp == l] = self.label_colours[l][0]
        g[temp == l] = self.label_colours[l][1]
        b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((c, 3, h, w)).astype(np.uint8)
        rgb[:, 0, :, :] = r
        rgb[:, 1, :, :] = g
        rgb[:, 2, :, :] = b
        return rgb
