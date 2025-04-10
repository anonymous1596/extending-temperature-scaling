#!/bin/bash

for model in resnet152 seresnet152 xception densenet161 inceptionresnetv2
do
  for dataset in cifar svhn lsun
  do
    python get_dropout_preds.py $model $dataset 200
  done
done