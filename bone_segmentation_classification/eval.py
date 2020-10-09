import torch
import torch.nn as nn

import numpy as np
from sklearn import metrics
from utils.metrics import dice_coefficient


def eval_net(net, data_loader, loss_function, pred=False, to_collect=None):
    with torch.no_grad():

        epoch_loss = [0] * len(loss_function)

        """ collected items """
        all_out = []
        all_features = []
        all_label = []

        for i, x in enumerate(data_loader):

            imgs = x[1]
            labels = x[2]
            output = net(imgs)  # to model

            """ calculate loss """
            loss_c, loss_all = loss_function(output, labels)
            for l in range(len(loss_all)):
                epoch_loss[l] += loss_all[l].item()

            """ collect output WRONG"""
            if pred:
                all_label.append(labels.cpu())
                all_out.append(output[0].cpu().detach())

        """ WRONG"""
        if pred:
            all_out = torch.cat(all_out, 0)
            all_label = torch.cat(all_label, 0)

        auc = dice_coefficient(x=all_out, y=all_label, n_channels=all_out.shape[1])
        """ Loss overall"""
        epoch_loss = [x / i for x in epoch_loss]

        return epoch_loss, auc,


