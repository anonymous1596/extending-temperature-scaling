import torch
import os
import pandas as pd
import sys
from recalibration_functions import get_TS
from eval_functions import get_ood_metrics_entropy
from pathlib import Path
import seaborn as sns
sns.set()

seed = sys.argv[2]
torch.random.manual_seed(seed)
idcs = torch.load('splits/' + str(seed) + '.pt')
model = sys.argv[1]
setting = sys.argv[3]
ood_dataset = sys.argv[4]
path = 'results/' + setting + '/' + model + '/' + seed
if not(os.path.isfile(path + '/metrics/ood_entropy_' + ood_dataset + '.csv')):
  Path(path).mkdir(exist_ok=True, parents=True)
  func = getattr(__import__("recalibration_functions"), setting)
  val_size = 8000
  all_logits, all_labels = torch.load('../dropout_predictions/datasets/cifar/dropouts/' + model + '-150-regular' + '.pt')
  all_logits = all_logits.movedim(0,-1)
  test_logits, test_labels= all_logits[idcs[val_size:]], all_labels[idcs[val_size:]]
  ood_logits, _ = torch.load('../dropout_predictions/datasets/' + ood_dataset + '/dropouts/' + model + '-150-regular' + '.pt')
  ood_logits = ood_logits.movedim(0,-1)
  ood_logits = ood_logits[val_size:]
  res = []
  conf = pd.read_csv('config.csv', index_col = [0,1])

  base_ood_logits = get_TS(func, 'cpu')(ood_logits)
  TS = torch.load(path + '/maps/TS.pt', weights_only=False)
  metrics = get_ood_metrics_entropy(TS(ood_logits), base_ood_logits, TS(test_logits))
  res.append(pd.DataFrame([metrics], columns = ['AUC', 'Percent Change', 'Difference'], index = ['TS']))

  SP1 = torch.load(path + '/maps/SP1.pt', weights_only=False)
  metrics = get_ood_metrics_entropy(SP1(ood_logits), base_ood_logits, SP1(test_logits))
  res.append(pd.DataFrame([metrics], columns = ['AUC', 'Percent Change', 'Difference'], index = ['SP1']))

  SPTS = torch.load(path + '/maps/SPTS.pt', weights_only=False)
  metrics = get_ood_metrics_entropy(SPTS(ood_logits), base_ood_logits, SPTS(test_logits))
  res.append(pd.DataFrame([metrics], columns = ['AUC', 'Percent Change', 'Difference'], index = ['SPTS']))

  SPU = torch.load(path + '/maps/SPU.pt', weights_only=False)
  metrics = get_ood_metrics_entropy(SPU(ood_logits), base_ood_logits, SPU(test_logits))
  res.append(pd.DataFrame([metrics], columns = ['AUC', 'Percent Change', 'Difference'], index = ['SPU']))

  pd.concat(res).to_csv(path + '/metrics/ood_entropy_' + ood_dataset + '.csv')