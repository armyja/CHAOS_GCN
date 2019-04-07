import argparse

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
    parser = argparse.ArgumentParser()
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
    image_ID = ['' for l in range(slices)]
    slice_ID = ['' for l in range(slices)]
    average = np.zeros(slices)
    pixels = np.zeros((slices, args.organ_number), dtype=np.int)
    range_dict = {}
    for i in ['InPhase', 'OutPhase', 'T2SPIR']:
        range_dict[i] = [[9999.0, 0.0], [9999.0, 0.0], [9999.0, 0.0], [9999.0, 0.0]]
    # self.pixels[l] = int(s[organ_id + 5 - 1])  # sum of label
    for l in range(slices):
        s = slice_list[l].split(' ')
        image_ID[l] = s[0]
        slice_ID[l] = s[1]
        image_filename[l] = s[2]  # important
        label_filename[l] = s[3]  # important
        mode = ''
        for i in ['InPhase', 'OutPhase', 'T2SPIR']:
            if i in image_filename[l]:
                mode = i
        image1 = dcm2npy(image_filename[l]).astype(np.float32)
        label1 = png2npy(label_filename[l])
        if label1.sum() > 0:
            for j in range(args.organ_number):
                actives = image1[is_organ(label1, j+1)]
                no_actives = image1[label1 != (j + 1)]
                if len(actives) >0:
                    max_in_min = max(no_actives.min(), actives.min())
                    min_in_max = min(no_actives.max(), actives.max())

                    print('image: {}. slice: {}, mode: {}, organ: {}, organ_min: {}, organ_max:{}, min:{}, max:{}'.format(image_ID[l], slice_ID[l], mode, j+1, actives.min(), actives.max(), max_in_min, min_in_max))
                    range_dict[mode][j][0] = min(range_dict[mode][j][0], max_in_min)
                    range_dict[mode][j][1] = max(range_dict[mode][j][1], min_in_max)
    for i in ['InPhase', 'OutPhase', 'T2SPIR']:
        print('{}: {}'.format(i, range_dict[i]))
        #  sums
    #     s[organ_id + 5 - 1]
    # dcm =


# T1 OUT, T1 IN, T2

#


if __name__ == '__main__':
    main()
