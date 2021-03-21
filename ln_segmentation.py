import time, os
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_config import *
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from engine.lightning_classification import LitClassification
from pytorch_lightning.callbacks import ModelCheckpoint


def merge_ages(args, args_d, args_m):
    args.update(args_m)
    args.update(args_d)
    return args


def args_train():
    parser = OptionParser()
    parser.add_option('--prj', dest='prj', default='dess_segmentation',
                      type=str, help='name of the project')
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16,
                      type='int', help='batch size')
    parser.add_option('--bu', '--batch-update', dest='batch_update', default=64,
                      type='int', help='batch to update')
    parser.add_option('--lr', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda'),
    parser.add_option("--sv", action="store_true", dest='save_cp', default=False, help='save model parameters'),
    parser.add_option('-p', '--pred', action='store_true', dest='pred', default=False, help='only evaluate model')
    parser.add_option('-l', '--load', dest='load',
                      default=False, help='load saved model parameters')
    parser.add_option('--par', dest='parallel', action="store_true", help='run in multiple gpus')
    parser.add_option('--et', '--eval-train', dest='eval_train',
                      default=False, help='do training set evaluation')
    parser.add_option('-w', '--weight-decay', dest='weight_decay', default=0.0005,
                      type='float', help='weight decay')
    parser.add_option('--ini', '--ini-file', dest='ini_file', default='latest',
                      type=str, help='name of the ini file')
    parser.add_option('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
    parser.add_option('--mode', type=str, default='dummy')
    parser.add_option('--port', type=str, default='dummy')
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":

    args = vars(args_train())
    args['parallel'] = 'True'
    args['dir_checkpoint'] = '/home/ghc/Dataset/OAI_DESS_segmentation/checkpoints/'

    args_d = {'mask_name': 'bone_resize_B_crop_00',
              'data_path': os.getenv("HOME") + '/Dataset/OAI_DESS_segmentation/',
              'scale': 0.5,
              'mask_used':  [['femur'], ['tibia'], [1], [2, 3]],
              'interval': 4,
              'thickness': 0,
              'copy_channel': True,
              'pick': False,
              'method': 'automatic',
              'subject_splitting': '00',
              'val_percent': 0,
              'use_v01': False}

    args_m = {'backbone': 'vgg11',
              'depth': 5}

    args = merge_ages(args, args_d, args_m)

    """ split range"""
    def imorphics_split():
        train_00 = list(range(10, 71))
        eval_00 = list(range(1, 10)) + list(range(71, 89))
        train_01 = list(range(10+88, 71+88))
        eval_01 = list(range(1+88, 10+88)) + list(range(71+88, 89+88))
        return train_00, eval_00, train_01, eval_01

    train_00, eval_00, train_01, eval_01 = imorphics_split()

    # datasets
    from loaders.loader_imorphics import LoaderImorphics
    train_set = LoaderImorphics(args_d, subjects_list=train_00, transform=None)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=16, drop_last=False)
    eval_set = LoaderImorphics(args_d, subjects_list=eval_00, transform=None)
    eval_loader = DataLoader(eval_set, batch_size=args['batch_size'], shuffle=True, num_workers=16, drop_last=False)
    print(len(train_set))
    print(len(eval_set))

    """ Imports """
    from models.unet import UNet_clean
    net = UNet_clean(output_ch=len(args_d['mask_used']) + 1, backbone=args_m['backbone'], depth=args_m['depth'])
    from utils.metrics_segmentation import SegmentationCrossEntropyLoss, SegmentationDiceCoefficient
    loss_function = SegmentationCrossEntropyLoss()
    metrics = SegmentationDiceCoefficient()

    """ cuda """
    if args['legacy']:
        net = net.cuda()
        net = nn.DataParallel(net)
    net.par_freeze = []

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/' + args['prj'] + '/',
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    """ Lightning """
    ln_classification = LitClassification(args=args,
                                          train_loader=train_loader,
                                          eval_loader=eval_loader,
                                          net=net,
                                          loss_function=loss_function,
                                          metrics=metrics)
    if args['legacy']:
        ln_classification.overall_loop()
    else:
        tb_logger = pl_loggers.TensorBoardLogger('logs/' + args['prj'] + '/')
        trainer = pl.Trainer(gpus=4, accelerator='ddp',
                             max_epochs=100, progress_bar_refresh_rate=20, logger=tb_logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(ln_classification, train_loader, eval_loader)


# CUDA_VISIBLE_DEVICES=0,1,2,3 python ln_segmentation.py -b 16 --bu 64 --lr 0.01

