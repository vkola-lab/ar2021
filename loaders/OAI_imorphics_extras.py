import os, glob
import pandas as pd
import numpy as np
import torch
from itertools import compress


def imorphics_info():
    df = pd.read_excel(os.getenv("HOME") + '/Dropbox/Z_DL/OAIDataBase/oai_active_contour_parameters.xlsx')
    df['subject_id'] = df['subject_id'].values.astype(np.int32)
    return df


def moaks00():
    return pd.read_csv(os.getenv("HOME") + '/Dropbox/Z_DL/OAIDataBase/moaks00.csv')


def locate_lateral_and_medial(slices, pick=False):
    """ Select certain slices by the z-locations"""
    dess_info = oai_dess_info()
    for i in list(slices.keys()):
        loc = dess_info.loc[dess_info['ID'] == i]
        if pick == 'lateral':
            slices[i] = [x for x in slices[i] if int(x.split('_')[1]) in list(range(loc['lateral0'].values[0], loc['lateral1'].values[0] + 1))]
        if pick == 'medial':
            slices[i] = [x for x in slices[i] if int(x.split('_')[1]) in list(range(loc['medial0'].values[0], loc['medial1'].values[0] + 1))]
    return slices


def slice_location_normalized(dess_info, id, scheme=2):
    """ slices location information"""
    i = int(id.split('_')[0])
    s = int(id.split('_')[1])

    side, s_0, s_1, lateral0, lateral1, medial0, medial1 \
        = dess_info.loc[dess_info['ID'] == i,
                        ['side', 's_0', 's_1', 'lateral0', 'lateral1', 'medial0', 'medial1']].values[0]

    if scheme == 1:
        if side == 2:
            return (s - medial0) / (lateral1 - medial0)
        elif side == 1:
            return (s - medial1) / (lateral0 - medial1)
    # lateral---notch---medial
    # 0       1       2      3
    elif scheme == 2:
        if side == 2:
            if s >= lateral0:
                return (s - s_1) / (lateral0 - s_1)
            elif s >= medial1:
                return (s - lateral0) / (medial1 - lateral0) + 1
            #elif s >= s_0:
            #    return (s - medial1) / (s_0 - medial1) + 2
            else:
                return (s - medial1) / (s_0 - medial1) + 2
        elif side == 1:
            if s <= lateral1:
                return (s - s_0) / (lateral1 - s_0)
            elif s <= medial0:
                return (s - lateral1) / (medial0 - lateral1) + 1
            #elif s <= s_1:
            #    return (s - medial0) / (s_1 - medial0) + 2
            else:
                return (s - medial0) / (s_1 - medial0) + 2


def get_ids_by_subjects(dir_img, method):
    neighbors = dict()
    slices = dict()
    dess_info = imorphics_info()

    """Returns a list of the ids in the directory"""
    if method == 'automatic':
        ids = list(set([f.split('/')[-1].split('.')[0] for f in glob.glob(dir_img+'*')]))
        ids = [tuple(int(y) for y in (x.split('_'))) for x in ids]
        ids = sorted(ids, key=lambda element: [e for e in element])

        slices_all = dict()
        for x in ids:
            if x[0] not in slices_all:
                slices_all[x[0]] = list()
            slices_all[x[0]].append('_'.join([str(y) for y in x]))

        subjects = list(slices_all.keys())
        subjects.sort()

        for s in subjects:
            slices[s] = slices_all[s]

    if method == 'manual':
        subjects = list(dess_info['ID'])
        for s in subjects:
            slice_range = list(range(dess_info.loc[dess_info['ID'] == s, 's_0'].values[0],
                                   dess_info.loc[dess_info['ID'] == s, 's_1'].values[0] + 4, 4))
            slices[s] = [str(s) + '_' + str(y) for y in slice_range]
            for x in slices[s]:
                neighbors[x] = [x]

    return subjects, slices


def imorphics_dess_info():
    dess_info = oai_dess_info()
    subjects = [(dess_info.iloc[i]['subject_id'], dess_info.iloc[i]['tags'] + '_imorphics') for i in
                range(len(dess_info['ID']))]
    crop = {'subjects': subjects,
            'cropx0': dess_info['cropx0'], 'cropx1': dess_info['cropx1'],
            'cropy0': dess_info['cropy0'], 'cropy1': dess_info['cropy1']}
    crop = pd.DataFrame(crop)

    zrange = {'subjects': subjects, 's_0': dess_info['s_0'], 's_1': dess_info['s_1']}
    zrange = pd.DataFrame(zrange)
    return subjects, crop, zrange


def oai_mri_info(dir_img, dir_tag, crop_file):
    subjects = glob.glob(dir_img + dir_tag + '/*')
    subjects.sort()
    subjects = [(int(x.split(dir_img + dir_tag + '/')[1].split('.')[0]), dir_tag) for x in subjects]
    if crop_file is not None:
        crop_file['subjects'] = subjects
        crop = crop_file[['subjects', 'cropx0', 'cropx1', 'cropy0', 'cropy1']]
        zrange = crop_file[['subjects', 's_0', 's_1']]
    else:
        crop = None
        zrange = None
    return subjects, crop, zrange


def dess_crop_files(x):
    df = pd.read_csv(x, header=None)
    df.columns = ['ID', 's_0', 's_1', 'cropx0', 'cropx1', 'cropy0', 'cropy1']
    df['ID'] = df['ID'].astype(np.int32)
    return df


def get_oai_XR(subjects, tags):
    """ labels"""
    df = pd.read_csv('/home/ghc/Dropbox/Z_DL/scripts/OAI_stats/collected/XR.csv')

    y = dict()
    y['subjects'] = subjects

    for tag in tags:
        label = []
        for x in subjects:
            try:
                label.append(df.loc[(df['ID'] == x[0]) & (df['SIDE'] == 2), tag].values[0])
            except:
                label.append(np.nan)
        y[tag] = np.array(label)

    labels = pd.DataFrame(y)
    labels = labels.loc[~labels.isnull().any(axis=1)]

    return labels




