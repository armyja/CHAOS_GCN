import torch

import visdom

from utils import *

def main():

    vis = visdom.Visdom()
    vis.delete_env(env=f'main_')
    viz = visdom.Visdom(env=f'main_')
    image_list = open('/media/jeffrey/D/CHAOS/lists/training.txt', 'r').read().splitlines()
    min = 0
    max = 1000
    for i in range(len(image_list)):
        if min > i or max < i:
            continue
        s = image_list[i].split(' ')
        image_filename = s[2]
        label_filename = s[3]

        image = np.load(image_filename)
        image = np.repeat(image.reshape(1, 1, image.shape[0], image.shape[1]), 3, axis=1).astype(np.int32)
        image = torch.from_numpy(image) / 4
        label = np.load(label_filename)
        label = np.repeat(label.reshape(1, 1, label.shape[0], label.shape[1]), 3, axis=1).astype(np.int32)
        label = torch.from_numpy(label)

        viz.images(image, opts=dict(title=f'', caption=''))
        viz.images(label, opts=dict(title=f'', caption=''))

if __name__ == '__main__':
    main()