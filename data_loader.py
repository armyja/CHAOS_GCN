import math

import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
import os

from utils.utils import *


class OrganSeg(Dataset):
    def __init__(self, current_fold, list_path, organ_id, slice_threshold=1, transforms=None):
        self.organ_ID = organ_id
        self.transforms = transforms
        image_list = open(training_set_filename(list_path, current_fold), 'r').read().splitlines()

        self.training_image_set = np.zeros((len(image_list)), dtype=np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.training_image_set[i] = int(s[0])

        slice_list = open(list_training(list_path), 'r').read().splitlines()
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

        self.active_index = [l for l, p in enumerate(self.pixels)
                             if self.image_ID[l] in self.training_image_set]  # true active

    def __getitem__(self, index):
        # stuff
        image_path = 'Image/Train/Images/'
        mask_path = 'Mask/Train/Images/'
        image = Image.open(image_path + self.list[index])
        image = image.convert('RGB')
        mask = Image.open(mask_path + self.list[index])
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (image, mask)

    def __len__(self):
        return len(self.list)  # of how many data(images?) you have


class LungSegTest(Dataset):
    def __init__(self, path='Image/Test/Images', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)

        self.transforms = transforms

    def __getitem__(self, index):
        # stuff
        image_path = 'Image/Test/Images/'
        mask_path = 'Mask/Test/Images/'
        image = Image.open(image_path + self.list[index])
        image = image.convert('RGB')
        mask = Image.open(mask_path + self.list[index])
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (image, mask)

    def __len__(self):
        return len(self.list)
