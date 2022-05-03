# Define a function for training process, returning training (test) loss and errors
import os
import re
import time
import scipy.stats
import torch
import torch.nn as nn

def weights_init(m):   
    nn.init.xavier_uniform_(m.weight.data)


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


def train_model(loaders, model, criterion, optimizer, log_saver,model_name, use_gpu=True, num_epochs=100):
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
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    steps += 1

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
        
        if (epoch+1) % 50 == 0 or epoch == num_epochs - 1:
            print('Saving for %sth model..'%(epoch+1))
            state = {'net': model, 'epoch': epoch, 'log': log_saver}
            cur_model_name = model_name+'_%sth.cpt'%(epoch+1)
            cur_model_name = os.path.join('checkpoints',cur_model_name)
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            # torch.save(state, './checkpoint_CNN/ckpt_epoch_{}.best'.format(epoch))
            torch.save(state, cur_model_name)
        
        print('current best test accuracy:',best_acc)

        # if epoch>2:
        #     if log_saver['test_loss'][-1]>log_saver['test_loss'][-2] and log_saver['test_loss'][-2]>log_saver['test_loss'][-3]:
        #         print('meet early stop condition! stop!')
        #         break

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    )

    print('best test accuracy:',best_acc)
    return model, log_saver

