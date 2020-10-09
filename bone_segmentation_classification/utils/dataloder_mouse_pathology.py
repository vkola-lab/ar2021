import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os, glob, shutil
import numpy as np
from PIL import Image
from skimage import measure
import scipy.ndimage.morphology as morphology
from utils.data_utils import *


def get_roi_from_mask(x, dis=100):
    y = (morphology.distance_transform_edt(1 - (x == x.max())) <= dis)
    return np.nonzero(y.sum(1))[0][0], np.nonzero(y.sum(0))[0][0],\
           np.nonzero(y.sum(1))[0][-1], np.nonzero(y.sum(0))[0][-1]


def draw_boxes_on(x, boxes):
    x0 = x/x.max()
    x0 = np.concatenate([np.expand_dims(x0, 2)] * 3, 2)
    for i in range(boxes.shape[0]):
        b = boxes[i, :]
        x0[b[0]:b[2] + 1, b[1], 0] = 1
        x0[b[0]:b[2] + 1, b[3], 0] = 1
        x0[b[0], b[1]:b[3] + 1, 0] = 1
        x0[b[2], b[1]:b[3] + 1, 0] = 1

    imagesc(x0)


class MakeMousePatch():
    def __init__(self, raw_img, raw_mask, args_d, mode):
        self.args_d = args_d
        self.mode = mode
        self.raw_img = raw_img
        self.raw_mask = raw_mask

    def make_random_patches(self, size=64, n=100, min_area=20):

        self.remake_dir()

        img = self.raw_img
        mask = self.raw_mask
        img_patches = []
        mask_patches = []
        i = 0
        while i < n:
            coor = (np.random.randint(img.shape[0] - size), np.random.randint(img.shape[1] - size))
            mask_patch = mask[coor[0]:coor[0] + size, coor[1]:coor[1] + size]
            if mask_patch.sum() > min_area:
                img_patches.append(np.expand_dims(img[coor[0]:coor[0] + size, coor[1]:coor[1] + size], 2))
                mask_patches.append(np.expand_dims(mask_patch, 2))
                i += 1

        img_patches = np.concatenate(img_patches, 2)
        mask_patches = np.concatenate(mask_patches, 2)

        self.save_patches(img_patches, mask_patches)

    def remake_dir(self):
        if self.mode == 'train':
            dir_list = ['dir_train_imgs', 'dir_train_masks']
        if self.mode == 'eval':
            dir_list = ['dir_eval_imgs', 'dir_eval_masks']
        dir_list = [self.args_d['dir_data'] + self.args_d[x] for x in dir_list]
        for d in dir_list:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.mkdir(d)

    def save_patches(self, img_patches, mask_patches):
        for i in range(img_patches.shape[2]):
            img = Image.fromarray(img_patches[:, :, i])
            mask = Image.fromarray(mask_patches[:, :, i])
            if self.mode == 'train':
                img.save(self.args_d['dir_data'] + self.args_d['dir_train_imgs'] + str(i) + '.png')
                mask.save(self.args_d['dir_data'] + self.args_d['dir_train_masks'] + str(i) + '.png')
            elif self.mode == 'eval':
                img.save(self.args_d['dir_data'] + self.args_d['dir_eval_imgs'] + str(i) + '.png')
                mask.save(self.args_d['dir_data'] + self.args_d['dir_eval_masks'] + str(i) + '.png')


class MousePatchDataset(Dataset):
    def __init__(self, args_d, mode):

        self.copy_channel = args_d['copy_channel']

        # Initialize the mode of dataset and decide the path to th folders
        self.mode = mode  # train or val
        if self.mode == 'train':
            self.dir_img = args_d['dir_data'] + args_d['dir_train_imgs']
            self.dir_mask = args_d['dir_data'] + args_d['dir_train_masks']
        elif self.mode == 'eval':
            self.dir_img = args_d['dir_data'] + args_d['dir_eval_imgs']
            self.dir_mask = args_d['dir_data'] + args_d['dir_eval_masks']

        # scan the img containing folder WRONG
        id_list = np.array([x.split(self.dir_img)[1] for x in glob.glob(self.dir_img + '*')])
        id_list = id_list[np.argsort([int(x.split('.')[0]) for x in id_list])]
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def get_bounding_boxes(self, mask_individual):
        boxes = []
        if len(mask_individual) > 0:
            for i in range(mask_individual.shape[0]):
                a_mask = mask_individual[i, :, :]
                x0 = np.nonzero(a_mask)[0].min()
                x1 = np.nonzero(a_mask)[0].max()
                y0 = np.nonzero(a_mask)[1].min()
                y1 = np.nonzero(a_mask)[1].max()
                boxes.append((x0, y0, x1, y1))

            boxes = np.concatenate([np.expand_dims(x, 0) for x in boxes], 0)

        return boxes

    def get_individual_masks(self, mask):
        blobs = measure.label(mask)
        individual_masks = []
        for i in range(1, blobs.max() + 1):
            individual_masks.append(np.expand_dims(blobs == i, 0))

        if len(individual_masks) > 0:
            individual_masks = np.concatenate(individual_masks, 0)
        return individual_masks

    def load_imgs(self, id):
        img = Image.open(self.dir_img + id)
        img = np.array(img)
        img = img / img.max()
        img = img.astype(np.float32)
        if self.copy_channel:
            img = np.concatenate([np.expand_dims(img, 0)] * 3, 0)
        return img

    def load_masks(self, id):
        mask = Image.open(self.dir_mask + id)
        mask = np.array(mask)
        mask = mask.astype(np.int32)

        mask = (mask == 255)

        return mask

    def __getitem__(self, idx):
        id = self.id_list[idx]
        mask = self.load_masks(id)
        individual_masks = self.get_individual_masks(mask)
        boxes = self.get_bounding_boxes(individual_masks)

        target = {}
        target['boxes'] = boxes
        target['masks'] = individual_masks
        target['labels'] = np.ones(len(boxes))

        return self.load_imgs(id), target


args_d = {'dir_data': 'data/mouse_pathology/',
          'dir_train_imgs': 'train_imgs/',
          'dir_train_masks': 'train_masks/',
          'dir_eval_imgs': 'eval_imgs/',
          'dir_eval_masks': 'eval_masks/',
          'copy_channel': True}

dir_data = 'data/mouse_pathology/source/'
raw_img = np.uint8(np.array(Image.open(dir_data + 'example_img.tiff')))
raw_mask = np.uint8(np.array(Image.open(dir_data + 'example_mask.tiff')))

# rough cropping
c = get_roi_from_mask(raw_mask, dis=50)
raw_img = raw_img[c[0]:c[2], c[1]:c[3]]
raw_mask = raw_mask[c[0]:c[2], c[1]:c[3]]

# create random patches
H = raw_mask.shape[0]
H_divide = H//3 * 2
train_patches = MakeMousePatch(raw_img=raw_img[:H_divide, :], raw_mask=raw_mask[:H_divide, :], args_d=args_d, mode='train')
train_patches.make_random_patches(size=32, n=500, min_area=50)

eval_patches = MakeMousePatch(raw_img=raw_img[H_divide:, :], raw_mask=raw_mask[H_divide:, :], args_d=args_d, mode='eval')
eval_patches.make_random_patches(size=32, n=200, min_area=50)

# dataset
ds = MousePatchDataset(args_d, 'eval')
for i in range(30):
    img, target = ds.__getitem__(i)
    print(len(target['boxes']))
    draw_boxes_on(img.sum(0), target['boxes'])

# plot one case
img, target = ds.__getitem__(i)
draw_boxes_on(img.sum(0), target['boxes'])
