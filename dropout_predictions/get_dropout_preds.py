from torch.utils.data import DataLoader, TensorDataset
import importlib
import torch
from tqdm import tqdm
from pathlib import Path
import json
import sys
sys.path.append(str(Path(f"{__file__}").parent.parent))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_logits(model, valid_loader):
  model.train()
  logits_list = []
  labels_list = []
  with torch.no_grad():
      for input, label in valid_loader:
          input, label = input.to(device), label.to(device)

          input = input
          logits = model(input)
          logits_list.append(logits.detach().cpu())
          labels_list.append(label.detach().cpu())
      logits = torch.cat(logits_list)
      labels = torch.cat(labels_list)
  return logits, labels

def get_dropout_preds(model, valid_loader, dropouts = 100):
  res = [get_logits(model,valid_loader) for j in tqdm(range(dropouts))]
  res1 = torch.stack([res[i][0] for i in range(len(res))])
  labels = res[0][1]
  return (res1, labels)

torch.random.manual_seed(10)
batch_size = 64

trained_folder = '../models/trained_final'

model_name = sys.argv[1]
dataset = sys.argv[2]
n_dropouts = int(sys.argv[3])
model_dict = json.load(open('model_dict.json'))

file_name = model_dict[model_name]['file_name']
function_name = model_dict[model_name]['function_name']

Path('datasets/' + dataset + '/dropouts').mkdir(exist_ok=True, parents=True)
model_name += '-150-regular'
print('Model: ' + str(model_name))
model = importlib.import_module('models.' + file_name, package = './').__getattribute__(function_name)()
a = torch.load(trained_folder + '/' + model_name + '.pth', map_location = device)
model.load_state_dict(a)
model = model.to(device)

x, y = torch.load('datasets/' + dataset + '/tensor_data.pt')
dl = DataLoader(TensorDataset(x, y), batch_size = batch_size, shuffle = False)
logits, labels = get_dropout_preds(model, dl, n_dropouts)
torch.save((logits, labels), 'datasets/' + dataset  + '/dropouts/' + model_name + '.pt')

