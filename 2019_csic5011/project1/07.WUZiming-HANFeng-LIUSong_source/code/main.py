import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from dataloader import DataLoader
from torch.autograd import Variable
from dice_loss import dice_coeff

from eval import eval_net
import resnet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, get_train_data, get_test_data

def get_train_data(input_path, target_path):
    return DataLoader(input_path, target_path, 'train')

def get_test_data(input_path, target_path):
    return DataLoader(input_path, target_path, 'test')

def train_net(net,
              train_loader,
              val_loader,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              img_scale=0.5,
              args = None):


    #dir_mask = 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'

    #ids = get_ids(dir_img)
    #ids = split_ids(ids)

    #iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_loader),
               len(val_loader), str(save_cp), str(gpu)))

#    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        if epoch != 0:
            
            print('Starting epoch {}/{}.'.format(epoch, epochs))
            net.train()

            # reset the generators
            #train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
            #val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

            epoch_loss = 0

            for i, bat in enumerate(train_loader, 1):
                optimizer.zero_grad()
                imgs, true_masks = bat[0], bat[1]


                #imgs = np.array([i[0] for i in b]).astype(np.float32)
                #true_masks = np.array([i[1] for i in b])

                #imgs = torch.from_numpy(imgs)
                #true_masks = torch.from_numpy(true_masks)

                if gpu:
                    imgs = Variable(imgs.cuda()).float()
                    true_masks = Variable(true_masks.cuda()).float()
                else:
                    imgs = Variable(imgs).float()
                    true_masks = Variable(true_masks).float()


                masks_pred = net(imgs)
                masks_probs_flat = masks_pred.view(-1)

                true_masks_flat = true_masks.view(-1)

                loss = criterion(masks_probs_flat, true_masks_flat)
                epoch_loss += loss.item()
                if i%args.step == 0:
                    print('Epoch: {} -- {} --- loss: {}'.format(epoch, i * batch_size / 9000, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch {} finished ! Loss: {}'.format(epoch, epoch_loss / i))

        if 1:
            net.eval()
            tot = 0
            pixel_loss = 0


            #val_list = []
            #val_gt_list = []



            for i, bat in enumerate(val_loader, 1):            
                imgs, true_masks = bat[0], bat[1]

                if gpu:
                    imgs = Variable(imgs.cuda()).float()
                    true_masks = Variable(true_masks.cuda()).float()
                else:
                    imgs = Variable(imgs).float()
                    true_masks = Variable(true_masks).float()


                #imgs, true_masks = bat[0].to(device=device, dtype=torch.float), bat[1].to(device=device, dtype = torch.float)
                #imgs = imgs.unsqueeze(0)
                #true_masks = true_masks.unsqueeze(0)
                mask_pred = net(imgs)
                #mask_pred_filter = (mask_pred > 0.5).float()
                #tot += dice_coeff(mask_pred_filter, true_masks).item()
                masks_probs_flat = mask_pred.view(-1)
                true_masks_flat = true_masks.view(-1)
                loss = criterion(masks_probs_flat, true_masks_flat)
                pixel_loss += loss.item()
                #mask_array = mask_pred.data.cpu().numpy()
                #gt_array = true_masks.data.cpu().numpy()
                #val_list.append(mask_array)
                #val_gt_list.append(gt_array)
            #val_dice = tot / (i+1)  
            #val_dice = eval_net(net, val, gpu)
            #print('Validation Dice Coeff: {}'.format(val_dice))
            print('Pixel Loss is : {}'.format(pixel_loss/i))


            #np.save("val_pred_data.npy", val_list)
            #np.save("val_gt_data.npy", val_gt_list)
            #return 
        if epoch%args.freq == 0:
            torch.save(net.state_dict(),
                       dir_checkpoint + '0402CP{}.pth'.format(epoch ))
            print('Checkpoint {} saved !'.format(epoch ))

def predit(net,
          train_loader,
          val_loader,
          epochs=5,
          batch_size=1,
          lr=0.1,
          val_percent=0.05,
          save_cp=True,
          gpu=True,
          img_scale=0.5,
          args = None):
        criterion = nn.BCELoss()
        net.eval()
        tot = 0
        pixel_loss = 0

        print("Running prediction")
        val_list = []
        val_gt_list = []



        for i, bat in enumerate(val_loader, 1):            
            imgs, true_masks = bat[0], bat[1]

            if gpu:
                imgs = Variable(imgs.cuda()).float()
                true_masks = Variable(true_masks.cuda()).float()
            else:
                imgs = Variable(imgs).float()
                true_masks = Variable(true_masks).float()


            #imgs, true_masks = bat[0].to(device=device, dtype=torch.float), bat[1].to(device=device, dtype = torch.float)
            #imgs = imgs.unsqueeze(0)
            #true_masks = true_masks.unsqueeze(0)
            mask_pred = net(imgs)
            mask_pred_filter = (mask_pred > 0.5).float()
            #tot += dice_coeff(mask_pred_filter, true_masks).item()
            masks_probs_flat = mask_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)
            pixel_loss += loss.item()
            #mask_array = mask_pred.data.cpu().numpy()
            #gt_array = true_masks.data.cpu().numpy()
            #val_list.append(mask_array)
            #val_gt_list.append(gt_array)
        #val_dice = tot / (i+1)  
        #val_dice = eval_net(net, val, gpu)
        #print('Validation Dice Coeff: {}'.format(val_dice))
        print('Pixel Loss is : {}'.format(pixel_loss/i))


        #np.save("./save_data/val_pred_data_0402_1.npy", val_list)
        #np.save("./save_data/val_gt_data_0402_1.npy", val_gt_list)
        #print("Data saved!")
        return 


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=500, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=1, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=1, help='downscaling factor of the images')
    parser.add_option('-t', '--step', dest='step', type='int',
                      default=10, help='print step')
    parser.add_option('-f', '--frequency', dest='freq', type='int',
                      default=20, help='frequency to save checkpoints')

    parser.add_option('-r', '--predit', dest='pred', type='int',
                      default=1, help='run prefiction')

    (options, args) = parser.parse_args()
    return options

cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(123)

device = torch.device("cuda" if 1 else "cpu")
print(device)
if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(1)
    net = reset.resnet18(n_channels=1, n_classes=2)

    #args.load = "checkpoints/0402CP120.pth"
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=args.gpu).cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    input_path = '../data/train_feature.npy'
    target_path = '../data/train_social_label.npy'

    train_set = get_train_data(input_path, target_path)
    test_set = get_test_data(input_path, target_path)


    train_data_loader = DataLoader(dataset = train_set, num_workers=4, batch_size=args.batchsize, shuffle=True)
    test_data_loader = DataLoader(dataset = test_set, num_workers=4, batch_size=args.batchsize, shuffle=False)

    
    # predit(net=net,
    #           train_loader = train_data_loader,
    #           val_loader = test_data_loader,
    #           epochs=args.epochs,
    #           batch_size=args.batchsize,
    #           lr=args.lr,
    #           gpu=args.gpu,
    #           img_scale=1, 
    #           args =args)

    try:
        train_net(net=net,
                  train_loader = train_data_loader,
                  val_loader = test_data_loader,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=1, 
                  args =args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
