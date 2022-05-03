# Define a function for training process, returning training (test) loss and errors
import os
import re
import time
import scipy.stats
import torch
import torch.nn as nn
from urllib3 import Retry
import sys
from scipy.stats import pearsonr, spearmanr
import numpy as np


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() #
    def flush(self):
        self.log.flush()

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise ValueError('only allow input True or False!')

def str2tuple(v):
    if isinstance(v,tuple):
        return v
    else:
        v = v.split(',')
        return tuple([int(e) for e in v])

def calc_correlation(x,y):
    x, y = np.array(x), np.array(y)
    spe_corre = spearmanr(x, y)
    pea_corre = pearsonr(x, y)
    return spe_corre, pea_corre


def calc_sharp_ratio(x):
    x = np.array(x)
    return np.mean(x)/np.std(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_model(loaders, model, criterion, optimizer, log_saver,model_name, regular_store, device, num_epochs=100):
    since = time.time()
    steps = 0
    best_acc = 0
    best_model_name = os.path.join('checkpoints',model_name+'_best.cpt')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:

            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            margin_error_meter = AverageMeter()

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            for i, data in enumerate(loaders[phase]):
                inputs, labels = data
                # if use_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                # print('outputs',outputs[:10])
                # print('x',inputs[0])
                # print('y',labels[0])
                # print('loss',loss)

                if phase == 'train':
                    loss.backward()
                    # for name, parms in model.named_parameters(): 
                    #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                    #     ' -->grad_value:',parms.grad)
                    optimizer.step()
                    steps += 1
                
                # time.sleep(5)
                N = outputs.size(0)

                loss_meter.update(loss.data.item(), N)
                acc_meter.update(
                    accuracy(outputs.data, labels.data)[-1].item(), N)

            epoch_acc = acc_meter.avg / 100
            epoch_loss = loss_meter.avg
            epoch_error = 1 - acc_meter.avg / 100

            if phase == 'train':
                log_saver['train_loss'].append(epoch_loss)
                log_saver['train_error'].append(epoch_error)

            elif phase == 'test':

                log_saver['test_loss'].append(epoch_loss)
                log_saver['test_error'].append(epoch_error)

            print(
                f'{phase} loss: {epoch_loss:.4f}; accuracy: {epoch_acc:.4f}'
            )

        # if epoch % 30 == 0 or epoch == num_epochs - 1:
        if epoch_acc>best_acc:
            best_acc = epoch_acc
            print('Saving for best model..')
            state = {'net': model, 'epoch': epoch, 'log': log_saver}

            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            # torch.save(state, './checkpoint_CNN/ckpt_epoch_{}.best'.format(epoch))
            torch.save(state, best_model_name)
            
        if regular_store:
            if (epoch+1) % 50 == 0 or epoch == num_epochs - 1:
                print('Saving for %sth model..'%(epoch+1))
                state = {'net': model, 'epoch': epoch, 'log': log_saver}
                cur_model_name = os.path.join('checkpoints',model_name+'_%sth.cpt'%(epoch+1))
                if not os.path.isdir('checkpoints'):
                    os.mkdir('checkpoints')
                # torch.save(state, './checkpoint_CNN/ckpt_epoch_{}.best'.format(epoch))
                torch.save(state, cur_model_name)
        else:
            if epoch>2:
                if log_saver['test_loss'][-1]>log_saver['test_loss'][-2] and log_saver['test_loss'][-2]>log_saver['test_loss'][-3]:
                # if log_saver['test_error'][-1]>log_saver['test_error'][-2] and log_saver['test_error'][-2]>log_saver['test_error'][-3]:
                    print('meet early stop condition! stop!')
                    break

        print('current best test accuracy:',best_acc)



    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    )

    print('best test accuracy:',best_acc)
    return model, log_saver


def test_model(loaders, model, criterion, optimizer, log_saver,model_name, regular_store, device, num_epochs=100):
    since = time.time()
    steps = 0
    pred_labels = []
    true_lables = []
    # true_scores = []
    scores = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['test']:

            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            margin_error_meter = AverageMeter()

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            for i, data in enumerate(loaders[phase]):
                inputs, labels = data
                # if use_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                outputs_softmax = torch.softmax(outputs,dim=1)
                # scores.extend(list(outputs_softmax.type(torch.FloatTensor).detach().cpu().numpy()))

                # scores.extend([outputs_softmax[m,n] for m, n in enumerate(labels)])

                _, preds = torch.max(outputs.data, 1)
                hard_labels = torch.argmax(outputs,dim=1)
                pred_labels.extend(list(hard_labels.type(torch.FloatTensor).detach().cpu().numpy()))
                true_lables.extend(list(labels.type(torch.FloatTensor).detach().cpu().numpy()))
                # true_scores.extend(list(labels.type(torch.FloatTensor).detach().cpu().numpy()))

                # scores.extend(torch.gather(outputs_softmax, 0, hard_labels.view(-1,1)).detach().cpu().numpy())
                scores.extend(outputs_softmax[:,1].detach().cpu().numpy())

                # print('pred_labels',pred_labels)
                # print('true_lables',true_lables)
                # time.sleep(10)
                loss = criterion(outputs, labels)

                N = outputs.size(0)

                loss_meter.update(loss.data.item(), N)
                acc_meter.update(
                    accuracy(outputs.data, labels.data)[-1].item(), N)

            epoch_acc = acc_meter.avg / 100
            epoch_loss = loss_meter.avg
            epoch_error = 1 - acc_meter.avg / 100

            if phase == 'train':
                log_saver['train_loss'].append(epoch_loss)
                log_saver['train_error'].append(epoch_error)

            elif phase == 'test':

                log_saver['test_loss'].append(epoch_loss)
                log_saver['test_error'].append(epoch_error)

            print(
                f'{phase} loss: {epoch_loss:.4f}; accuracy: {epoch_acc:.4f}'
            )
            

    return model, log_saver, pred_labels, true_lables, scores


def pearson_correlations(x, y):
    """
    Pearson相关系数

    :param x: 向量x
    :param y: 向量y
    :return rho: (Pearson)相关系数
    """
    cov = np.cov(x, y, bias=True)[0][1]  # 有偏估计, 当样本长度确定时
    # cov = np.cov(x, y)[0][1]  # 无偏估计
    std_x = np.std(x)
    std_y = np.std(y)
    
    rho = cov / (std_x * std_y)
    
    return rho


def spearmans_rank_correlation(x, y):
    """
    Spearman相关系数

    :param x: 向量x
    :param y: 向量y
    :return rho: spearman相关系数
    """
    # 合并向量
    # [[-6, 8, -4, 10],
    #  [-7, -5, 7, 9]]
    spearman_matrix = np.vstack((x, y))

    # 得到排序后的x和y的rank值(因为x与y同型)
    # [1, 2, 3, 4]
    rank = np.arange(1, len(x) + 1)

    # 将x的rank值向量合并到矩阵上
    # [[-6, -4, 8, 10],
    #  [-7, 7, -5, 9],
    #  [1, 2, 3, 4]]
    spearman_matrix = spearman_matrix[:, spearman_matrix[0].argsort()]
    spearman_matrix = np.vstack((spearman_matrix, rank))

    # 将y的rank值向量合并到矩阵上
    # [[-6, 8, -4, 10],
    #  [-7, -5, 7, 9],
    #  [1, 3, 2, 4],
    #  [1, 2, 3, 4]]
    spearman_matrix = spearman_matrix[:, spearman_matrix[1].argsort()]
    spearman_matrix = np.vstack((spearman_matrix, rank))

    # 重新按照x的rank值排列
    # [[-6, -4, 8, 10],
    #  [-7, 7, -5, 9],
    #  [1, 2, 3, 4],
    #  [1, 3, 2, 4]]
    spearman_matrix = spearman_matrix[:, spearman_matrix[0].argsort()]

    # 求squa(d)
    # [0, 1, 1, 0]
    d_square = (spearman_matrix[2] - spearman_matrix[3]) ** 2

    rho = 1 - (6 * sum(d_square) / (len(x) * (len(x) ** 2 - 1)))

    return rho