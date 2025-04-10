import torch
import pandas as pd
import sys
from eval_functions import get_calibration_metrics
from pathlib import Path
import seaborn as sns
import os
sns.set()

seed = sys.argv[2]
torch.random.manual_seed(seed)
idcs = torch.load('splits/' + str(seed) + '.pt')
model = sys.argv[1]
setting = sys.argv[3]
path = 'results/' + setting + '/' + model + '/' + seed
if not(os.path.isfile(path + '/metrics/calibration.csv')):
  Path(path + '/metrics').mkdir(parents=True, exist_ok=True)

  func = getattr(__import__("recalibration_functions"), setting)

  val_size = 8000
  all_logits, all_labels = torch.load('../dropout_predictions/datasets/cifar/dropouts/' + model + '-150-regular' + '.pt')
  all_logits = all_logits.movedim(0,-1)
  val_logits, val_labels= all_logits[idcs[0:val_size]], all_labels[idcs[0:val_size]]
  test_logits, test_labels= all_logits[idcs[val_size:]], all_labels[idcs[val_size:]]

  res = []
  conf = pd.read_csv('config.csv', index_col = [0,1])

  
  TS = torch.load(path + '/maps/TS.pt', weights_only=False)
  metrics = get_calibration_metrics(TS, test_logits, test_labels)
  res.append(pd.DataFrame([metrics], columns = ['NLL', 'ECE10', 'ECE15', 'ECE20'], index = ['TS']))

  SP1 = torch.load(path + '/maps/SP1.pt', weights_only=False)
  metrics = get_calibration_metrics(SP1, test_logits, test_labels)
  res.append(pd.DataFrame([metrics], columns = ['NLL', 'ECE10', 'ECE15', 'ECE20'], index = ['SP1']))

  SPTS = torch.load(path + '/maps/SPTS.pt', weights_only=False)
  metrics = get_calibration_metrics(SPTS, test_logits, test_labels)
  res.append(pd.DataFrame([metrics], columns = ['NLL', 'ECE10', 'ECE15', 'ECE20'], index = ['SPTS']))

  SPU = torch.load(path + '/maps/SPU.pt',weights_only=False)
  metrics = get_calibration_metrics(SPU, test_logits, test_labels)
  res.append(pd.DataFrame([metrics], columns = ['NLL', 'ECE10', 'ECE15', 'ECE20'], index = ['SPU']))

  pd.concat(res).to_csv(path + '/metrics/calibration.csv')