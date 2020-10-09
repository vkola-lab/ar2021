import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os, glob
import numpy as np
from PIL import Image


class DataBone(Dataset):
    def __init__(self, args_d, mode):

        # Initialize the mode of dataset and decide the path to th folders
        self.mode = mode  # train or val
        if self.mode == 'train':
            self.dir_img = args_d['dir_data'] + args_d['dir_train_imgs']
            self.dir_mask = args_d['dir_data'] + args_d['dir_train_masks']
        elif self.mode == 'val':
            self.dir_img = args_d['dir_data'] + args_d['dir_eval_imgs']
            self.dir_mask = args_d['dir_data'] + args_d['dir_eval_masks']

        # scan the img containing folder
        id_list = [x.split(self.dir_img)[1] for x in glob.glob(self.dir_img + '*')]
        id_list.sort()
        self.id_list = id_list

    def load_imgs(self, id):
        img = Image.open(self.dir_img + id)
        img = np.array(img)
        if len(img.shape) == 3:
            img = img[:, :, 0]
        img = np.concatenate([np.expand_dims(img, 0)] * 3, 0)

        # normalize
        img = img / img.max()
        img = img.astype(np.float32)
        return img

    def load_masks(self, id):
        mask = Image.open(self.dir_mask + id)
        mask = np.array(mask)
        mask = mask.astype(np.int32)

        mask = (mask == 255)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        return mask

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        img = self.load_imgs(id)
        mask = self.load_masks(id)

        return id, img, mask


if __name__ == "__main__":

    args = {'batch_size': 16}

    args_d = {'dir_data': 'data/subchondral_patches/',
              'dir_train_imgs': 'train_imgs/',
              'dir_train_masks': 'train_masks/',
              'dir_eval_imgs': 'eval_imgs/',
              'dir_eval_masks': 'eval_masks/',
              'copy_channel': True}

    args_m = {'output_ch': 2,
              'backbone': 'vgg11',
              'depth': 5}

    train_set = DataBone(args_d, mode='train')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True)

    val_set = DataBone(args_d, mode='val')
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=2, drop_last=False)

    # Loading using data loader
    id, img, true_mask = train_set.__getitem__(100)

    # Loading using data loader
    for i, x in enumerate(train_loader):
        id, img, true_mask = x
        print(img.shape)

    """ Test Network """
    from models.unet import UNet_clean
    net = UNet_clean(output_ch=args_m['output_ch'],
                     backbone=args_m['backbone'],
                     depth=args_m['depth'])

    output = net(img)

    """ Test Loss Function """
    from utils.loss import SegmentationLoss
    loss_function = SegmentationLoss()
    loss, _ = loss_function(x=output, y=true_mask)
    print(loss)

