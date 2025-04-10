import torch
from torch.nn import Softmax, CrossEntropyLoss
from torch.distributions import Categorical
from torch import nn
from sklearn.metrics import roc_curve, auc
import numpy as np
import os

def compute_entropy(logits):
  probs = Softmax(1)(logits)
  return Categorical(probs = probs, validate_args=False).entropy()  

# Code for ECE taken from https://github.com/gpleiss/temperature_scaling
class _ECELoss(nn.Module):
    def __init__(self, n_bins=10, rank = 0):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins
        self.rank = rank
    def forward(self, logits, labels):
        softmaxes = Softmax(1)(logits)
        a = torch.sort(softmaxes, dim = 1, descending = True)
        confidences, predictions = a[0][:,self.rank], a[1][:,self.rank]
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if in_bin.float().sum() > 5:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.detach().cpu().item()

    
def compute_AUC(ood_logits, test_logits, ent_func):
  test_entropies = ent_func(test_logits)
  ood_entropies = ent_func(ood_logits)
  y_roc = np.hstack([np.zeros(len(test_logits)), np.ones(len(ood_logits))])
  y_pred_roc = np.hstack([test_entropies, ood_entropies])
  fpr, tpr, _ = roc_curve(y_roc, y_pred_roc, pos_label = 1)
  return auc(fpr, tpr)

def get_calibration_metrics(recalib_model, test_logits, test_labels):
  recalib_test_logits = recalib_model(test_logits)
  nll = CrossEntropyLoss()(recalib_test_logits, test_labels).item()
  ECE10 = _ECELoss(n_bins=10).forward(recalib_test_logits, test_labels)
  ECE15 = _ECELoss(n_bins=15).forward(recalib_test_logits, test_labels)
  ECE20 = _ECELoss(n_bins=20).forward(recalib_test_logits, test_labels)
  res = [nll, ECE10, ECE15, ECE20]
  return res

def get_decr(logits, logits_orig, ent_func):
  ents = ent_func(logits)
  ents_orig = ent_func(logits_orig)
  percent_decr = (ents - ents_orig < 0).float().mean().item()
  avg_ent_change = (ents.mean() - ents_orig.mean()).item()
  return percent_decr, avg_ent_change

def compute_conf(logits):
  probs = Softmax(1)(logits)
  return 1 - probs.max(1)[0]

def get_ood_metrics_entropy(map_ood_logits, ood_logits, map_test_logits):
  map_ood_logits = map_ood_logits.detach()
  ood_logits = ood_logits.detach()
  map_test_logits = map_test_logits.detach()
  AUC_ent_roc = compute_AUC(map_ood_logits, map_test_logits, compute_entropy)
  ent_decr, ent_change = get_decr(map_ood_logits, ood_logits, compute_entropy)
  return [AUC_ent_roc, ent_decr, ent_change]

def get_ood_metrics_varrat(map_ood_logits, ood_logits, map_test_logits):
  map_ood_logits = map_ood_logits.detach()
  ood_logits = ood_logits.detach()
  map_test_logits = map_test_logits.detach()
  AUC_ent_roc = compute_AUC(map_ood_logits, map_test_logits, compute_conf)
  ent_decr, ent_change = get_decr(map_ood_logits, ood_logits, compute_conf)
  return [AUC_ent_roc, ent_decr, ent_change]