#!/bin/bash

for i in $(seq 0 19);
do 
  for setting in pointpred compose locshift convert
  do
    for model in resnet152 seresnet152 xception densenet161 inceptionresnetv2
    do
      python get_recalibration_maps.py $model $i $setting
      python get_calibration_metrics.py $model $i $setting
      for ood_dataset in svhn lsun
      do
        python get_ood_metrics.py $model $i $setting $ood_dataset
      done
    done
  done
done   