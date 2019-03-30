from utils import utils
# choose organ

# Mapping
# 0 is background
# 1 is liver
# 2 is right kidney
# 3 is left kidney
# 4 is spleen
from utils import *


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

# get organ files
def parse_args():
    pass

def main():
    args = parse_args()
    list_path = list_path_from_root(args.root_dir)
    # get all image name
    slice_list = open(list_training_all(list_path), 'r').read().splitlines()
    slices = len(slice_list)
    # image_ID = np.zeros(slices, dtype=np.int)
    # slice_ID = np.zeros(slices, dtype=np.int)
    image_filename = ['' for l in range(slices)]
    label_filename = ['' for l in range(slices)]
    average = np.zeros(slices)
    pixels = np.zeros((slices, args.organ_number), dtype=np.int)
    # self.pixels[l] = int(s[organ_id + 5 - 1])  # sum of label
    for l in range(slices):
        s = slice_list[l].split(' ')
        # image_ID[l] = s[0]
        # slice_ID[l] = s[1]
        image_filename[l] = s[2]  # important
        label_filename[l] = s[3]  # important
        #  sums
    #     s[organ_id + 5 - 1]
    # dcm =

# T1 OUT, T1 IN, T2

#