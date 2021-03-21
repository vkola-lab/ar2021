import torch
import glob
from models.unet import UNet_clean
from collections import OrderedDict
from utils.data_utils import imagesc
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


def get_seg(checkpoint, dcm):
    state_dict = checkpoint['state_dict']
    state_dict_new = OrderedDict((k.split('net.')[1], v) for k, v in state_dict.items())

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


crop = pd.read_csv('/media/ghc/GHc_data1/dess_segmentation_pred/SAG_3D_DESS_LEFT_00.csv')
dcm_path = '/media/ghc/GHc_data1/OAI/extracted/SAG_3D_DESS_LEFT_00/'
subject = 9000099

crop_idx = crop.loc[crop['ID'] == subject].values[0, :]
dcm = get_dcm(dcm_path + str(subject) + '/')
dcm = dcm[crop_idx[1]:crop_idx[2], :, crop_idx[3]:crop_idx[4], crop_idx[5]:crop_idx[6]]

cartilage = get_seg(torch.load('checkpoints/dess_segmentation/cartilage.ckpt'), dcm)
bone = get_seg(torch.load('checkpoints/dess_segmentation/bone0.ckpt'), dcm)
combined = 0 * bone
combined[bone == 1] = 1
combined[bone == 2] = 2
combined[cartilage == 1] = 3
combined[cartilage == 2] = 4


