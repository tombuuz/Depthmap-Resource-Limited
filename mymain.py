import json
import math
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from nni.retiarii.oneshot.pytorch import ProxylessTrainer
import nni.retiarii.nn.pytorch as nn
import nni.retiarii.strategy as strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii import serialize
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment, debug_mutated_model

from mymodel import MobileNet
from mynyudata import NyuDataset
import myutils



class ComputeError(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = 0
        
    def forward(self, pred, target):
    	rmse = (target - pred) ** 2
    	rmse = torch.sqrt(rmse.mean())
    	return rmse


class Result(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = 0
        
    def forward(self, pred, target):

        valid_mask = ((target>0) + (pred>0)) > 0
        pred = 1e3 * pred[valid_mask]
        target = 1e3 * target[valid_mask]
        abs_diff = (pred - target).abs()
        mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        return torch.Tensor(rmse)

def accuracy(pred, target):

    valid_mask = ((target>0) + (pred>0)) > 0
    pred = 1e3 * pred[valid_mask]
    target = 1e3 * target[valid_mask]
    abs_diff = (pred - target).abs()
    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    res = dict()
    res["acc"] = rmse
    return res, res

    
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
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


if __name__ == '__main__':
	# Define a model
    base_model = MobileNet()

    # Create dataset
    valdir = os.path.join('..', 'nyudepthv2', 'train', 'office_0003')
    train_dataset = serialize(NyuDataset, root=valdir, train=True)


    # uncomment the following two lines to debug a generated model
    #debug_mutated_model(base_model, trainer, [])
    #exit(0)

    # Optimizer and Loss 
    momentum = 0.1
    nesterov = True
    optimizer = torch.optim.SGD(get_parameters(base_model), lr=0.05, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)


    # Trainer
    device = torch.device("cpu")
    trainer = ProxylessTrainer(base_model,
                           loss=ComputeError(),
                           dataset=train_dataset,
                           optimizer=optimizer,
                           metrics=lambda output, target: accuracy(output, target),
                           num_epochs=50,
                           log_frequency=10,
                           device=device)
    trainer.fit()
    print('Final architecture:', trainer.export())
    json.dump(trainer.export(), open('checkpoint.json', 'w'))
