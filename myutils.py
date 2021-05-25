import random

import json
import torch
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
import torch.nn.functional as F
from nni.retiarii import serialize
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment, debug_mutated_model
#from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import transforms
from mynyu import NyuDataset

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

    
def get_parameters(model, keys=None, mode='include'):
    if keys is None:
        for name, param in model.named_parameters():
            yield param
    elif mode == 'include':
        for name, param in model.named_parameters():
            flag = False
            for key in keys:
                if key in name:
                    flag = True
                    break
            if flag:
                yield param
    elif mode == 'exclude':
        for name, param in model.named_parameters():
            flag = True
            for key in keys:
                if key in name:
                    flag = False
                    break
            if flag:
                yield param
    else:
        raise ValueError('do not support: %s' % mode)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))

