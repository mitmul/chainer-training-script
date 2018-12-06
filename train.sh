#!/bin/bash

python train_imagenet_resnet50_blockmode.py \
train_random.txt \
val_random.txt \
--train_root /mnt/netapp_vol01/shunta/datasets/ILSVRC2015/Data/CLS-LOC/train \
--val_root /mnt/netapp_vol01/shunta/datasets/ILSVRC2015/Data/CLS-LOC/val \
--batchsize 128 \
--gpu 0 \
--arch resnet50 \
--epoch 90
