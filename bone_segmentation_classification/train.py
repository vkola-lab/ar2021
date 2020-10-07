import os, time
import torch
import torch.nn as nn
import numpy as np
from eval import eval_net


def train_net(args,
              train_loader,
              val_loader,
              net,
              loss_function,
              optimizer,
              epochs,
              save_cp,
              gpu,
              pred=False):

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Weight Decay: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, args['batch_size'], args['lr'], args['weight_decay'], len(train_loader.dataset),
               len(val_loader.dataset), str(save_cp), str(gpu)))

    if pred:
        epochs = 1

    for epoch in range(epochs):
        tini = time.time()

        """ Training """
        epoch_loss = [0] * len(loss_function)

        if not pred:
            net.train(mode=True)
            train_loader.dataset.mode = 'train'

            for i, x in enumerate(train_loader):

                # (id, tag), img, y = x
                imgs = x[1]
                labels = x[2]

                """ forward model"""
                output = net(imgs)  # to model
                loss = 0

                """ calculate loss """
                loss_c, loss_all = loss_function(output, labels)
                for l in range(len(loss_all)):
                    epoch_loss[l] += loss_all[l].item()

                """ Total Loss """
                loss += (loss_c * 1)

                """ Total Loss """
                loss.backward()

                """ Update """
                if i % (args['batch_update'] // args['batch_size']) == 0 or i == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

        else:
            """ STUPID """
            i = len(train_loader)

        """ Evaluation """
        net.train(mode=False)
        net.eval()

        val_loader.dataset.mode = 'val'
        val_out = eval_net(net, val_loader, loss_function, pred=True, to_collect=args['to_collect'])
        val_loss = val_out[0]
        val_acc = val_out[1]

        if args['eval_train']:
            val_loader.dataset.mode = 'train'
            train_out = eval_net(net, val_loader, loss_function)
        else:
            train_out = val_out

        train_loss = train_out[0]
        train_acc = train_out[1]

        if pred:
            print(train_out[2][2].shape)
            np.save('features_out.npy', train_out[2][2])

        """ print stats """
        print_out = {
                     'Epoch: {}': [epoch],
                     'Time: {:.2f} ': [time.time() - tini],
                     'Train Loss: ' + '{:.4f} ' * len(epoch_loss): [x / i for x in epoch_loss],
                     'Loss (T/V): ' + '{:.4f} ' * len(train_loss) * 2: train_loss + val_loss,
                     'Acc: ' + '{:.4f} ' * len(val_acc): val_acc,
                     }
        print(' '.join(print_out.keys()).format(*[j for i in print_out.values() for j in i]))

        if save_cp:
            try:
                torch.save(net.module.state_dict(),
                           (args['dir_checkpoint'] + args['ini_file'] + '_CP{}' + '.pth').format(epoch))
            except:
                torch.save(net.state_dict(),
                           (args['dir_checkpoint'] + args['ini_file'] + '_CP{}' + '.pth').format(epoch))
