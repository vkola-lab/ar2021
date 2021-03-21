import os, glob, torch, time
import numpy as np
from PIL import Image
import torch.nn as nn


def resampling(x, new_size):
    l = len(x.shape)
    if l == 2:
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)
    if l == 3:
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(1)

    x = nn.functional.interpolate(
        input=x, size=new_size, mode='bilinear', align_corners=True).numpy()

    return np.squeeze(x)


def mask_overlap(img, mask):
    img = np.concatenate([np.expand_dims(img, 2)]*3, 2)
    img[:,:,0] = np.multiply(img[:,:,0], 1 - mask)
    imagesc(img)


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


def append_dict(x):
    return [j for i in x for j in i]


def resize_and_crop(pilimg, scale):
    dx = 32

    w0 = pilimg.size[0]//dx * dx
    h0 = pilimg.size[1]//dx * dx
    pilimg = pilimg.crop((0, 0, w0, h0))

    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    img = pilimg.resize((newW, newH))

    return img


class imorphics_masks():
    def __init__(self, adapt=None):
        self.adapt = adapt

    def load_masks(self, id, dir, fmt, scale):
        if self.adapt is not None:
            id = str(self.adapt.index((int(id.split('/')[1]), id.split('/')[0])) + 1) + '_' + str(int(id.split('/')[2]))
        raw_masks = []
        for d in dir:
            temp = []
            for m in d:
                x = Image.open(os.path.join(m, id + fmt))  # PIL
                x = resize_and_crop(x, scale=scale)  # PIL
                x = np.array(x)  # np.int32
                temp.append(x.astype(np.float32))  # np.float32

            raw_masks.append(temp)

        out = np.expand_dims(self.assemble_masks(raw_masks), 0)
        return out

    def assemble_masks(self, raw_masks):
        converted_masks = np.zeros(raw_masks[0][0].shape, np.long)
        for i in range(len(raw_masks)):
            for j in range(len(raw_masks[i])):
                converted_masks[raw_masks[i][j] == 1] = i + 1

        return converted_masks


def load_masks(id, dir, fmt, scale):
    raw_masks = []
    for d in dir:
        temp = []
        for m in d:
            x = Image.open(os.path.join(m, id + fmt))  # PIL
            x = resize_and_crop(x, scale=scale)  # PIL
            x = np.array(x)  # np.int32
            temp.append(x.astype(np.float32))  # np.float32

        raw_masks.append(temp)

    return raw_masks


def assemble_masks(raw_masks):
    converted_masks = np.zeros(raw_masks[0][0].shape, np.long)
    for i in range(len(raw_masks)):
        for j in range(len(raw_masks[i])):
            converted_masks[raw_masks[i][j] == 1] = i + 1

    return converted_masks
