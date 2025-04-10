from torch.utils.data import DataLoader,TensorDataset
import torch
from torch.nn import Softmax, CrossEntropyLoss
from torch import nn
from torch import optim
from tqdm import tqdm

def optimize(map, logits, labels,
            n_epochs = 100, lr = 0.01):

    dl = DataLoader(TensorDataset(logits, labels), batch_size = len(logits), shuffle = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.Adam(map.parameters(), lr=lr)
    losses = torch.zeros(n_epochs)

    for i in tqdm(range(n_epochs)):
      for x,y in dl:
        x,y = x.to(device), y.to(device)
        loss = CrossEntropyLoss()(map(x), y)
        losses[i] += loss.detach().cpu().item()*len(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      losses[i] /= len(dl.dataset)
    return losses

def mean_shift(preds, new_means):
  diff = new_means - preds.mean(2)
  return preds + diff.unsqueeze(2).tile((1,1,preds.shape[2]))

def inv_softmax(probs):
  return probs.log() - probs.log().mean(1).unsqueeze(1).tile((1,probs.shape[1]))

def bayes_pred(logits):
  return inv_softmax(Softmax(1)(logits).mean(2))


############################################################
#                                                   
############################################################
class TemperatureScalingRecalibration(nn.Module):
  def __init__(self, func, T = 1):
    super().__init__()
    self.T = nn.Parameter(torch.ones(1)*T)
    self.func = func
  
  def map(self, x):
    return x*self.T
    
  def forward(self, x):
    return self.func(self.map, x)
  
def softplus(x, a, b, c, u):
  return -(c/a)*torch.log(u + torch.exp(-a*(x-b)))

class SoftplusRecalibration(nn.Module):
  def __init__(self, func, a = 1, b = 1, c = 1, u = 0.1):
    super().__init__()
    self.a = nn.Parameter(torch.ones(1)*a)
    self.b = nn.Parameter(torch.ones(1)*b)
    self.c = nn.Parameter(torch.ones(1)*c)
    self.u = nn.Parameter(torch.ones(1)*u)
    self.func = func

  def map(self, x):
    return softplus(x, self.a ** 2, self.b, self.c ** 2, self.u ** 2)

  def forward(self, x):
    return self.func(self.map, x)

def pointpred(map, x):
  return map(x[:,:,0])

def compose(map, x):
  return bayes_pred(map(x))

def locshift(map, x):
  return bayes_pred(mean_shift(x, map(x.mean(2))))

def convert(map, x):
  return map(bayes_pred(x))

def get_TS(func, device = None):
  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return TemperatureScalingRecalibration(func).to(device)

def get_SPTS(func, T):
  recalib_model = SoftplusRecalibration(func, c = T)
  recalib_model.c.requires_grad = False
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return recalib_model.to(device)
  
def get_SP1(func):
  recalib_model = SoftplusRecalibration(func)
  recalib_model.c.requires_grad = False
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return recalib_model.to(device)

def get_SPU(func):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return SoftplusRecalibration(func, a = 0.135).to(device)


class SoftplusRecalibrationShift(nn.Module):
  def __init__(self, a = 1, b = 1, c = 1, u = 0.1, w = torch.zeros(100)):
    super().__init__()
    self.a = nn.Parameter(torch.ones(1)*a)
    self.b = nn.Parameter(torch.ones(1)*b)
    self.c = nn.Parameter(torch.ones(1)*c)
    self.u = nn.Parameter(torch.ones(1)*u)
    self.w = nn.Parameter(w)

  def forward(self, x):
    x1 = x[:,:,0]
    shift = (x1 @ self.w).unsqueeze(1).tile(1,x.shape[1])
    return softplus(x1 + shift, self.a ** 2, self.b, self.c ** 2, self.u ** 2)
  
def get_SPTS_shift(T):
  recalib_model = SoftplusRecalibrationShift(c = T)
  recalib_model.c.requires_grad = False
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return recalib_model.to(device)
  
def get_SP1_shift():
  recalib_model = SoftplusRecalibrationShift()
  recalib_model.c.requires_grad = False
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return recalib_model.to(device)

def get_SPU_shift():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  return SoftplusRecalibrationShift(a = 0.135).to(device)
