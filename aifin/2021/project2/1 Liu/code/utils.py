import torch
import numpy as np
import random
import os
from copy import deepcopy
import math
import torch.nn as nn
import torch.nn.functional as F


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def pretrain(model, loader, optimizer, device):
    model.train()
    ## training with ce
    loss_avg = Averager() 
 #   print(len(loader))
    for batch_idx, batch in enumerate(loader):
  #      print(batch_idx)
        model.zero_grad()
        optimizer.zero_grad()
        train_inputs, train_targets = batch[0], batch[1]
        train_targets = train_targets.long()
        train_inputs = train_inputs.to(device=device)
        train_targets = train_targets.to(device=device)
        train_logits = model(train_inputs)
        loss = nn.CrossEntropyLoss()(train_logits, train_targets)
        loss_avg.add(loss.item())
        loss.backward()
        optimizer.step()
    print("Train Loss %.4f" % (loss_avg.item()))
    return loss_avg.item()

def evaluate_batch(model, data_loader, device):
    model.eval()
    correct = num = 0
    for iter, pack in enumerate(data_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        targets = target.long()
        logits = model(data)
        _, pred = logits.max(1)
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
   # print('Correct : ', correct)
   # print('Num : ', num)
 #   print('Test ACC : ', correct / num)
    torch.cuda.empty_cache()
    model.train()
    return correct / num
