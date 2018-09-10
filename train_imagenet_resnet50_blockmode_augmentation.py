#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import random

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import cv2
from PIL import Image
import glob
import random
import skimage
import skimage.io as io
import skimage.transform
import sys
import numpy as np
import math
#from pylab import *
#from matplotlib import pyplot
#import matplotlib.image as mpimg

import resnet50


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, min_aspect_ratio_change, min_area_change, max_area_change, random = True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.min_aspect_ratio_change = min_aspect_ratio_change
        self.min_area_change =  min_area_change
        self.max_area_change = max_area_change
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

        image, label = self.base[i]
        ch, h, w = image.shape

        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        #if ch != 3:
        #   image = np.resize(image, (3, h, w))
        #    print("before:", image.shape)
        #    im = image#Image.open(image)
        #    image = im.convert('RGB')
        #    print("after:", image.shape)
        img_mean = np.ones((3, h, w), dtype=np.float32)
        for i in range(0, 3):
           if i == 0:
              mean_value = 104.0
           elif i == 1:
              mean_value = 117.0
           elif i == 2:
              mean_value = 123.0
           else:
              print("unexpected parameter")
           img_mean[i] *= mean_value
        #mean_values_R = np.ones((h, w), dtype=np.float32)* 104
        #mean_values_G = np.ones((h, w), dtype=np.float32)* 117
        #mean_values_B = np.ones((h, w), dtype=np.float32)* 123
        image -=img_mean
        #image -= self.mean
        image *= (1.0 / 255.0)  # Scale to [0, 1]

        if self.random:
            #image shape change to H,W,C
            image = image.swapaxes(1, 0).swapaxes(2, 1)
            random_flip = random.random()
            if random_flip >= 0.5:
                imgMirror = np.fliplr(image)
            else:
                imgMirror = image
            area = imgMirror.shape[0] * imgMirror.shape[1]
            i=0
            while i < 10:
                i = i+1
                aspect_ratio_change = random.uniform(self.min_aspect_ratio_change, 1/self.min_aspect_ratio_change)
                area_ratio = random.uniform(self.min_area_change, self.max_area_change)
                new_area = area * area_ratio
                new_height = int(math.sqrt(new_area) * aspect_ratio_change)
                new_width = int(math.sqrt(new_area) / aspect_ratio_change)
                if random.randint(0, 1):
                #if new_height < new_width:
                    new_height, new_width = new_width, new_height
                if new_height <= imgMirror.shape[0] and new_width <= imgMirror.shape[1]:
                #if crop_size <= new_height and  crop_size <= new_width:
                #imgScaled = skimage.transform.resize(imgMirror, (new_height, new_width))
                #start_y = random.randint(0, int(imgScaled.shape[0] - corp_size))
                #start_x = random.randint(0, int(imgScaled.shape[1] - crop_szie))
                    start_y = imgMirror.shape[0]- new_height + 1
                    start_x = imgMirror.shape[1]- new_width + 1
                    output_img = imgMirror[start_y : start_y + new_height, start_x : start_x + new_width]
                    image = skimage.transform.resize(output_img, (crop_size, crop_size))
                    image = image.swapaxes(1, 2).swapaxes(0, 1)
                    #return image
                else:
                #new_aspect_ratio_change = (float(new_width) / float(new_height))
                #new_height = int(float(crop_size)/ float(new_aspect_ratio_change))
                    #     """when height is larger"""
                #new_width = max(new_width, crop_size)
                    image = skimage.transform.resize(imgMirror, (crop_size, crop_size))
                #start_y = random.randint(0, max(0, int(imgScaled.shape[0] - crop_size)))
                #output_img = imgScaled[start_y:start_y+crop_size, 0:crop_size]
                    image = image.swapaxes(1, 2).swapaxes(0, 1)
                 #print ("else:" + str(output_img.shape))
                    #return image

            # Randomly crop a region and flip the image
            #top = random.randint(0, h - crop_size - 1)
            #left = random.randint(0, w - crop_size - 1)
            #if random.randint(0, 1):
            #    image = image[:, :, ::-1]
        else:
            # Crop the center
            image = image[:, top:bottom, left:right]
        image = image.astype(np.float32)
        #image -= self.mean[:, top:bottom, left:right]
        #image *= (1.0 / 255.0)  # Scale to [0, 1]
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

    # Initialize the model to train
    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()
    else:
        model.to_intel64()
    # Load the datasets and mean file
    mean = np.load(args.mean)

    min_aspect_ratio_change = 0.75
    min_area_change = 0.08
    max_area_change = 1.0
    train = PreprocessedDataset(
        args.train, args.train_root, mean, model.insize, min_aspect_ratio_change, min_area_change, max_area_change)
    val = PreprocessedDataset(
        args.val, args.val_root, mean, model.insize, min_aspect_ratio_change, min_area_change, max_area_change, False)
    '''
    train = PreprocessedDataset(
        args.train, args.train_root, mean, model.insize)
    val = PreprocessedDataset(
        args.val, args.val_root, mean, model.insize, False)
    '''
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 5000), 'iteration'
    log_interval = (10 if args.test else 100), 'iteration'

    trainer.extend(extensions.polynomial_shift.PolynomialShift('lr', 1, 900000),
                   trigger=(1, 'iteration'))
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
