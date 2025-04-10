from torchvision.datasets import SVHN, LSUN, CIFAR100
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torch
from pathlib import Path

transform = transforms.Compose(
    [
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ]
)

path = 'datasets/cifar' 
Path(path).mkdir(exist_ok=True, parents=True)
cifar = CIFAR100(root = 'data2', train = False, download = True,
            transform=transform)
cifar_dl = DataLoader(Subset(cifar, range(10000)), 
                    batch_size=10000, shuffle=False)
cifar_x, cifar_y = next(iter(cifar_dl))

torch.save((cifar_x, cifar_y), path + '/tensor_data.pt')

path = 'datasets/lsun'
Path(path).mkdir(exist_ok=True, parents=True)
lsun = LSUN(root = 'data2', classes = ['classroom_train'], 
            transform=transform)
lsun_dl = DataLoader(Subset(lsun, range(10000)), 
                    batch_size=10000, shuffle=False)
lsun_x, lsun_y = next(iter(lsun_dl))
torch.save((lsun_x, lsun_y), path + '/tensor_data.pt')

path = 'datasets/svhn'
Path(path).mkdir(exist_ok=True, parents=True)
svhn = SVHN(root='data2', split = 'train',
                            download=True, transform=transform)
svhn_dl = DataLoader(Subset(svhn, range(10000)), 
                    batch_size=10000, shuffle=False)
svhn_x, svhn_y = next(iter(svhn_dl))
torch.save((svhn_x, svhn_y), path + '/tensor_data.pt')
