import os, glob, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loaders.OAI_imorphics_extras import get_ids_by_subjects, imorphics_info
from loaders.loader_utils import imorphics_masks, append_dict, resize_and_crop, imagesc


class LoaderImorphicsDual(Dataset):
    def __init__(self, args_d, subjects_list, transform):
        self.Imorphics0 = LoaderImorphics(args_d, subjects_list[0], transform)
        # do not apply interval to the second dataset for matching
        args_d_01 = args_d.copy()
        args_d_01['interval'] = 1
        self.Imorphics1 = LoaderImorphics(args_d_01, subjects_list[1], transform)
        self.imorphics_info = imorphics_info()

    def __len__(self):
        return len(self.Imorphics0)

    def __getitem__(self, idx):
        id0, img0, (mask0,) = self.Imorphics0.__getitem__(idx)
        #id1, img1, (mask1,) = self.Imorphics1.__getitem__(idx)
        # find matching slices between 00 and 01
        i = int(id0.split('_')[0])
        s = int(id0.split('_')[1])
        id1_match = str(i+88) + '_' + str(int(s + self.imorphics_info['s_00_01'][i-1]))
        id1, img1, (mask1,) = self.Imorphics1.__getitem__(self.Imorphics1.ids.index(id1_match))

        img = np.concatenate([np.expand_dims(x, 0) for x in [img0, img1]], 0)
        mask = np.concatenate([np.expand_dims(x, 0) for x in [mask0, mask1]], 0)

        return (id0, id1), img, (mask, )#((mask0,), (mask1,))


class LoaderImorphics(Dataset):
    def __init__(self, args_d, subjects_list, transform):
        dir_img = os.path.join(args_d['data_path'], args_d['mask_name'], 'original/')
        dir_mask = [[os.path.join(args_d['data_path'], args_d['mask_name'],
                                  'train_masks/' + str(y) + '/') for y in x] for x in
                    args_d['mask_used']]

        self.transform = transform
        self.dir_img = dir_img
        self.fmt_img = glob.glob(self.dir_img+'/*')[0].split('.')[-1]
        self.dir_mask = dir_mask

        self.scale = args_d['scale']
        self.copy_channel = args_d['copy_channel']

        """ Masks """
        self.masks = imorphics_masks(adapt=None)

        """ Splitting Subjects"""
        self.subjects, self.slices = get_ids_by_subjects(self.dir_img, args_d['method'])
        self.ids = append_dict([self.slices[s][0:None:args_d['interval']] for s in subjects_list])

    def load_imgs(self, id):
        x = Image.open(self.dir_img + id + '.' + self.fmt_img)
        x = resize_and_crop(x, self.scale)
        x = np.expand_dims(np.array(x), 0)  # to numpy
        return x

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        # load image
        img = self.load_imgs(id)

        # load mask
        mask = self.masks.load_masks(id, self.dir_mask, '.png', scale=self.scale)

        # transform images
        if self.transform:
            img = self.transform(img)

        # normalization
        img = torch.from_numpy(img)
        img = img.type(torch.float32)
        img = img / img.max()

        if self.copy_channel:
            img = torch.cat([1*img, 1*img, 1*img], 0)

        return id, img, (mask, )



