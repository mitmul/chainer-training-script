
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function

import argparse
import os
import random

import chainer
import numpy as np
from chainer import training
from chainer.training import extensions
from chainercv import transforms
from chainercv import utils
import cv2 as cv

import resnet50


os.environ["CHAINER_TYPE_CHECK"] = "0"


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f').mean(axis=(1, 2))[:, None, None]
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        path, int_label = self.base._pairs[i]
        full_path = os.path.join(self.base._root, path)

        assert os.path.exists(full_path)
        image = cv.imread(full_path, cv.IMREAD_COLOR)
        image = image.astype(np.float32)[:, :, ::-1].transpose(2, 0, 1)
        label = np.array(int_label, dtype=np.int32)

        _, h, w = image.shape
        if h < crop_size or w < crop_size:
            image = transforms.scale(image, crop_size)

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1) if h > crop_size else 0
            left = random.randint(0, w - crop_size - 1) if w > crop_size else 0
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2 if h > crop_size else 0
            left = (w - crop_size) // 2 if w > crop_size else 0
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean
        image *= 0.0078125  # -128 ~ 128

        assert image.shape == (3, crop_size, crop_size), '{}: {}'.format(full_path, image.shape)

        return image, label


def main():
    archs = {
        'resnet50': resnet50.ResNet50,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='resnet50', help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=128,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=90,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--train_root', '-TR', default='.',
                        help='Root directory path of train image files')
    parser.add_argument('--val_root', '-VR', default='.',
                        help='Root directory path of val image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--val_interval', '-v', type=int, default=5000,
                            help='Validation Interval')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
    chainer.config.use_cudnn_tensor_core = 'auto'
    chainer.config.autotune = True
    chainer.config.cudnn_fast_batch_normalization = True

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = PreprocessedDataset(
        args.train, args.train_root, mean, 224)
    val = PreprocessedDataset(
        args.val, args.val_root, mean, 224, False)

    # for i in range(len(train)):
    #     print('{} / {}'.format(i, len(train)))
    #     assert os.path.exists(os.path.join(train.base._root, train.base._pairs[i][0]))
 
    for j in range(len(val)):
        print('{} / {}'.format(j, len(val)))
        val[j]
        assert os.path.exists(os.path.join(val.base._root, val.base._pairs[j][0]))


if __name__ == '__main__':
    main()
