Steps to perform the experiments from the paper:

1. First, we need to train the base models on CIFAR-100. We use the code from https://github.com/weiaicunzai/pytorch-cifar100. The only modifications we make are to the model functions. We need to add dropout layers to the models. Other than that, we change the number of epochs to 150, which is a setting in conf/global_setting.py. We put the trained models in the folder `models/trained_final`. 

2. Next, we obtain the dropout predictions. To do so, we need to download the datasets `CIFAR100`, `SVHN`, and the classroom split of `LSUN`. The first two can be downloaded from PyTorch, but for `LSUN`, we need to use the code from https://github.com/fyu/lsun. After downloading all three datasets, we put them in one folder named `data2` in the `dropout_predictions` folder and we run `get_ood_dataset_tensors.py` to convert them to `.pt` files. Then, we run `bash run.sh` to obtain the dropout predictions.

3. Next, we learn the recalibration mappings and compute the evaluation metrics by running `bash run.sh` from the `recalibration` folder.

4. Finally, we generate the plots by running `box_plots.ipynb` from the `visualize` folder.