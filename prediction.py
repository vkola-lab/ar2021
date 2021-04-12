import torch
import glob, os
from models.unet import UNet_clean
from collections import OrderedDict
from dess_utils.data_utils import imagesc
import numpy as np
import pandas as pd


def get_dcm(dcm_path):
    l = glob.glob(dcm_path + '*')
    l.sort()
    dcm = []
    for x in l:
        x = np.load(x)
        dcm.append(np.expand_dims(x, 0))
    dcm = np.concatenate(dcm, 0)
    dcm[dcm >= 400] = 400
    dcm = torch.from_numpy(dcm / dcm.max())
    dcm = torch.cat([dcm.unsqueeze(1)] * 3, 1)
    dcm = dcm.type(torch.FloatTensor)
    return dcm


def get_seg(checkpoint, dcm, lightning=False):
    if lightning:
        state_dict = checkpoint['state_dict']
        state_dict_new = OrderedDict((k.split('net.')[1], v) for k, v in state_dict.items())
    else:
        state_dict_new = checkpoint

    # network
    net = UNet_clean(output_ch=state_dict_new['Conv.weight'].shape[0], backbone='vgg11', depth=5)
    net.load_state_dict(state_dict_new)
    net.cuda()
    seg = []
    for i in range(dcm.shape[0]):
        x = dcm[i:i + 1, ::].cuda()
        y, = net(x)
        y = y.cpu().detach().numpy()
        y = np.argmax(y, 1)
        seg.append(y)
    seg = np.concatenate(seg, 0)
    return seg


if __name__ == '__main__':
    crop = pd.read_csv('data/testing/SAG_3D_DESS_LEFT_00.csv')
    dcm_path = 'data/testing/SAG_3D_DESS_LEFT/'
    if not os.path.isdir(dcm_path[:-1] + '_seg/'):
        os.mkdir(dcm_path[:-1] + '_seg/')

    subjects = glob.glob(dcm_path + '*')

    for s in subjects:
        subject = s.split('/')[-1]
        crop_idx = crop.loc[crop['ID'] == int(subject)].values[0, :]
        dcm = get_dcm(dcm_path + str(subject) + '/')
        dcm = dcm[crop_idx[1]:crop_idx[2], :, crop_idx[3]:crop_idx[4], crop_idx[5]:crop_idx[6]]
        seg = get_seg(torch.load('checkpoints/clean_femur_tibia_cartilage.pth'), dcm).astype(np.uint8)
        np.save(dcm_path[:-1] + '_seg/' + subject + '.npy', seg)


