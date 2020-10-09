import os, time, glob, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import DataBone

from models.unet import UNet_clean
from utils.loss import *

from train import train_net

from utils.make_config import *


def main(net, args, train_set, val_set, loss_function):
    for param in net.par_freeze:
        param.requires_grad = False
    if args['gpu']:
        net.cuda()

    """ data loader """
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=16, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=16, drop_last=False)

    optimizer = torch.optim.SGD(list(set(net.parameters()) - set(net.par_freeze)),
                                lr=args['lr'],
                                momentum=0.9,
                                weight_decay=args['weight_decay'])

    """ check points """
    if args['load']:
        net.load_state_dict(torch.load(args['dir_checkpoint'] + args['ini_file']
                                       + '_CP{}.pth'.format(args['load'])))
        print('Model loaded from {}'.format(args['dir_checkpoint'] + args['ini_file']
                                            + '_CP{}.pth'.format(args['load'])))
    if args['parallel']:
        print('run in multiple gpu')
        net = nn.DataParallel(net)

    """ Begin Training"""
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))
    train_net(args=args,
              train_loader=train_loader,
              val_loader=val_loader,
              net=net,
              loss_function=loss_function,
              optimizer=optimizer,
              epochs=args['epochs'],
              save_cp=args['save_cp'],
              gpu=args['gpu'],
              pred=args['pred'])


def args_train():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16,
                      type='int', help='batch size')
    parser.add_option('--bu', '--batch-update', dest='batch_update', default=16,
                      type='int', help='batch to update')
    parser.add_option('--lr', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda'),
    parser.add_option("--sv", action="store_true", dest='save_cp', default=False, help='save model parameters'),
    parser.add_option('-p', '--pred', action='store_true', dest='pred', default=False, help='only evaluate model')
    parser.add_option('-l', '--load', dest='load',
                      default=False, help='load saved model parameters')
    parser.add_option('--par', dest='parallel',
                      default=False, help='run in multiple gpus')
    parser.add_option('--et', '--eval-train', dest='eval_train',
                      default=False, help='do training set evaluation')
    parser.add_option('-w', '--weight-decay', dest='weight_decay', default=0.0005,
                      type='float', help='weight decay')
    parser.add_option('--ini', '--ini-file', dest='ini_file', default='latest',
                      type=str, help='name of the ini file')
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":

    args = vars(args_train())
    args['dir_checkpoint'] = '/home/ghc/Dataset/OAI_DESS_segmentation/checkpoints/'

    args_d = {'dir_data': 'data/mouse_pathology/',
              'dir_train_imgs': 'train_imgs/',
              'dir_train_masks': 'train_masks/',
              'dir_eval_imgs': 'eval_imgs/',
              'dir_eval_masks': 'eval_masks/',
              'copy_channel': True}

    args_m = {'n_classes': 2,
              'backbone': 'vgg11',
              'depth': 5}

    train_set = DataBone(args_d, mode='train')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    val_set = DataBone(args_d, mode='val')
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=2, drop_last=False)

    """ Model options """
    net = UNet_clean(output_ch=args_m['n_classes'], backbone=args_m['backbone'], depth=args_m['depth'])
    """ Loss function"""
    from utils.loss import SegmentationLoss
    loss_function = SegmentationLoss()

    """ Model extras """
    net.par_freeze = []#[y for x in [list(x.parameters()) for x in [getattr(net, 'encoder')]] for y in x]
    net.args_m = args_m

    """ output to collect"""
    args['to_collect'] = 'all_out.append(out.cpu().detach().numpy())'

    """ write config file and start training """
    write_config('config/' + args['ini_file'] + '.ini', {'train': args, 'model': args_m, 'data': args_d})

    main(net, args, train_set, val_set, loss_function)


