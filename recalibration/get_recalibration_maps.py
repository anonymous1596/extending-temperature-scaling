import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from recalibration_functions import get_SP1, get_SPTS, get_SPU, get_TS, optimize
from pathlib import Path
import seaborn as sns
sns.set()

seed = sys.argv[2]
torch.random.manual_seed(seed)
idcs = torch.load('splits/' + str(seed) + '.pt')
model = sys.argv[1]
setting = sys.argv[3]
path = 'results/' + setting + '/' + model + '/' + seed
Path(path + '/maps').mkdir(parents=True, exist_ok=True)
Path(path + '/losses').mkdir(parents=True, exist_ok=True)

func = getattr(__import__("recalibration_functions"), setting)

val_size = 8000
all_logits, all_labels = torch.load('../dropout_predictions/datasets/cifar/dropouts/' + model + '-150-regular' + '.pt')
all_logits = all_logits.movedim(0,-1)
val_logits, val_labels= all_logits[idcs[0:val_size]], all_labels[idcs[0:val_size]]
test_logits, test_labels= all_logits[idcs[val_size:]], all_labels[idcs[val_size:]]

res = []
conf = pd.read_csv('config.csv', index_col = [0,1])

if not(os.path.isfile(path + '/maps/TS.pt')):
  TS = get_TS(func)
  epochs = conf.loc[(model, setting), 'TS']
  losses = optimize(TS, val_logits, val_labels, conf.loc[(model, setting), 'TS'])
  TS.to('cpu')
  plt.figure()
  plt.plot(losses)
  plt.savefig(path + '/losses/TS.png')
  plt.show()
  torch.save(TS, path + '/maps/TS.pt')

if not(os.path.isfile(path + '/maps/SP1.pt')):
  SP1 = get_SP1(func)
  epochs = conf.loc[(model, setting), 'SP-1']
  losses = optimize(SP1, val_logits, val_labels, epochs)
  SP1.to('cpu')
  plt.figure()
  plt.plot(losses)
  plt.savefig(path + '/losses/SP1.png')
  plt.show()
  torch.save(SP1, path + '/maps/SP1.pt')

if not(os.path.isfile(path + '/maps/SPTS.pt')):
  TS = torch.load(path + '/maps/TS.pt', weights_only=False)
  SPTS = get_SPTS(func, TS.T.detach().item() ** 0.5)
  epochs = conf.loc[(model, setting), 'SP-TS']
  losses = optimize(SPTS, val_logits, val_labels, epochs)
  SPTS.to('cpu')
  plt.figure()
  plt.plot(losses)
  plt.savefig(path + '/losses/SPTS.png')
  plt.show()
  torch.save(SPTS, path + '/maps/SPTS.pt')

if not(os.path.isfile(path + '/maps/SPU.pt')):
  SPU = get_SPU(func)
  epochs = conf.loc[(model, setting), 'SP-U']
  losses = optimize(SPU, val_logits, val_labels, epochs)
  SPU.to('cpu')
  plt.figure()
  plt.plot(losses)
  plt.savefig(path + '/losses/SPU.png')
  plt.show()
  torch.save(SPU, path + '/maps/SPU.pt')
  