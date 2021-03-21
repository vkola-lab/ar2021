import torch
import numpy as np
from PIL import Image
import os, time, glob
from torch.utils.data import Dataset
from loaders.OAI_subjects_labels import OAISubejctsLabels


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


class LoaderNPYDual(Dataset):
    def __init__(self, args_d, two_subjects, contrastive_labels):

        self.two_subjects = two_subjects
        self.contrastive_labels = contrastive_labels
        self.dataset0 = LoaderNPY(args_d, subjects=two_subjects[0])
        self.dataset1 = LoaderNPY(args_d, subjects=two_subjects[1])

    def __len__(self):
        # Two datasets should be the same length
        assert (len(self.dataset0) == len(self.dataset1))
        return len(self.dataset0)

    def __getitem__(self, idx):
        subject0, img0, label0 = self.dataset0.__getitem__(idx)
        subject1, img1, label1 = self.dataset1.__getitem__(idx)
        #label = np.concatenate([np.expand_dims(l, 0) for l in [label0, label1]])
        label = np.concatenate([label0, label1], 0)

        return (subject0, subject1), (img0, img1), (label, self.contrastive_labels.iloc[idx]['progress'])


class LoaderNPY(Dataset):
    def __init__(self, args_d, subjects):
        self.subjects = subjects

        self.img_dir = args_d['img_dir']
        self.interval = args_d['interval']
        self.copy_channel = args_d['copy_channel']
        self.mask_transfer = args_d['mask_transfer']
        self.load_mode = args_d['load_mode']

    def __len__(self):
        return len(self.subjects)

    def load_imgs(self, id, tag, load_mode):
        if (load_mode == 'npy_3D') or (load_mode == 'npy_2D'):
            img = np.load(self.img_dir + tag + '/' + str(id) + '.npy')
            img = img[::self.interval, :, :]

            # copy channels
            if self.copy_channel:
                img = np.concatenate([np.expand_dims(img, 1)] * 3, 1)
            else:
                img = np.expand_dims(img, 1)
            img = img.astype(np.float32)
            img_original = 1 * img
            mask_transfer = self.mask_transfer
            for m in mask_transfer.keys():
                img[img_original == m] = mask_transfer[m]

        elif load_mode == 'image':
            img = Image.open(self.img_dir + tag + '/' + str(id) + '.png')
            img = np.array(img).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))

        return img

    def convert_to(self, destination):
        """ save to destination """
        for i in range(len(self)):
            x = self.__getitem__(i)
            imagesc(x[1][0, ::], show=False, save=os.path.join(destination, str(x[0][0]) + '.png'))

    def __getitem__(self, idx):
        id = self.subjects.iloc[idx]['subjects']
        tag = self.subjects.iloc[idx]['tag']

        # image
        img = self.load_imgs(id, tag, self.load_mode)

        # label
        label = self.subjects.iloc[idx]['label']
        return (id, tag), img, (label, )


if __name__ == "__main__":
    args_d = {'img_dir': '/media/ghc/GHc_data1/dess_segmentation_processed/',
              'tags_used': ['SAG_3D_DESS_LEFT_00_2D', 'SAG_3D_DESS_RIGHT_00_2D'],
                                 #'SAG_3D_DESS_LEFT_01_2D', 'SAG_3D_DESS_RIGHT_01_2D'],
              'variables': ['V00XRJSM', 'V00XRJSL'],
              'scale': 1.0,
              'interval': 4,
              'copy_channel': False,
              'mask_transfer': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
              'load_mode': 'image'
              }

    if 1:
        oai_subjects = OAISubejctsLabels(source=args_d['img_dir'],
                                         tags_used=args_d['tags_used'])

        subjects_train, subjects_val = oai_subjects.train_XR_validation_MOAKS()
        subjects_train['label'] = 0
        # datasets
        train_set = LoaderNPY(args_d, subjects=subjects_train)
        val_set = LoaderNPY(args_d, subjects=subjects_val)

        for i in range(len(val_set)):
            print(i)
            tini = time.time()
            x = val_set.__getitem__(i)
            print(time.time() - tini)

        #train_set.convert_to('/media/ghc/GHc_data1/dess_segmentation_pred/SAG_3D_DESS_RIGHT_00_2D/')
        #val_set.convert_to('/media/ghc/GHc_data1/dess_segmentation_pred/SAG_3D_DESS_RIGHT_00_2D/')
