import torch
import os
seeds = torch.arange(20)
if not(os.path.exists('splits')):
  os.mkdir('splits')
for seed in seeds:
  torch.random.manual_seed(seed)
  idcs = torch.randperm(10000)
  torch.save(idcs, 'splits/' + str(seed.item()) + '.pt')
  