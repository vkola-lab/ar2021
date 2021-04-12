import glob, os
import numpy as np
from dess_utils.data_utils import imagesc
from skimage import measure
from scipy import ndimage


def convert_3D_to_2D(source, destination):
    l = glob.glob(source + '*.npy')[:]
    l.sort()
    if not os.path.isdir(destination):
        os.makedirs(destination)
    for name in l:
        x = np.load(name)
        r = (x == 3).mean(1)
        g = (x == 4).mean(1)
        b = (((x == 1) + (x == 2)).mean(1))
        y = np.zeros((224, 224, 3))
        y[:r.shape[0], :, 0] = r
        y[:r.shape[0], :, 1] = g
        #y[:r.shape[0], :, 2] = b
        imagesc(y, show=False, save=destination + name.split('/')[-1].split('.')[0] + '.png')


def clean_bone_blobs(x):
    bone = ((x == 1) + (x == 2))
    bone_blobs = measure.label(bone)
    bone_blobs_size = []
    for i in range(bone_blobs.max()):
        bone_blobs_size.append((bone_blobs == i).sum())
    bone_blobs_size = np.array(bone_blobs_size)
    background_index = np.argsort(bone_blobs_size)[-1]
    femur_index = np.argsort(bone_blobs_size)[-2]
    tibia_index = np.argsort(bone_blobs_size)[-3]
    background = (bone_blobs == background_index)
    femur = (bone_blobs == femur_index)
    tibia = (bone_blobs == tibia_index)
    print(id)
    ratio = (femur.sum()+tibia.sum()) / bone.sum()
    print('Extracted bone percentage = ' + str(ratio))
    if ratio <= 0.9:
        return 0

    y = 0 * x
    y[femur == 1] = 1
    y[tibia == 1] = 2
    y[x == 3] = 3
    y[x == 4] = 4

    return y


def sbl_3d(x):
    dist_to_femur = ndimage.distance_transform_edt(x != 1)
    dist_to_tibia = ndimage.distance_transform_edt(x != 2)
    dist_to_fc = ndimage.distance_transform_edt(x != 3)
    dist_to_tc = ndimage.distance_transform_edt(x != 3)
    femur_sbl = np.multiply((x == 1), (dist_to_fc <= 3)/1)
    tibia_sbl = np.multiply((x == 2), (dist_to_tc <= 3)/1)

    femur_sbl_distance = np.multiply((femur_sbl == 1)/1, dist_to_tibia)
    tibia_sbl_distance = np.multiply((tibia_sbl == 1)/1, dist_to_femur)
    y = np.concatenate([np.expand_dims(x, 3) for x in [femur_sbl, tibia_sbl]], 3)
    y = y.astype(np.uint8)
    return y


if __name__ == '__main__':
    source = 'data/testing/SAG_3D_DESS_LEFT_seg/'
    destination_clean = source.split('_seg/')[0] + '_clean/'
    destination_sbl = source.split('_seg/')[0] + '_sbl/'
    if not os.path.isdir(destination_clean):
        os.mkdir(destination_clean)
    if not os.path.isdir(destination_sbl):
        os.mkdir(destination_sbl)
    l = sorted(glob.glob(source+'*.npy'))
    for name in l:
        x = np.load(name)
        id = name.split(source)[1]
        y = clean_bone_blobs(x)
        z = sbl_3d(y)
        np.save(destination_clean + id, y.astype(np.uint8))
        np.save(destination_sbl + id, z.astype(np.uint8))





