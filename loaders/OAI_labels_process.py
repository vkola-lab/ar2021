import numpy as np


def JSN_one_hot(x, ver):
    x.loc[:, 'label'] = 0
    x.loc[(x[ver + 'XRJSM'] == 0) & (x[ver + 'XRJSL'] == 0), 'label'] = 0
    x.loc[(x[ver + 'XRJSM'] >= 1) & (x[ver + 'XRJSL'] == 0), 'label'] = 1
    x.loc[(x[ver + 'XRJSM'] == 0) & (x[ver + 'XRJSL'] >= 1), 'label'] = 0
    x.loc[(x[ver + 'XRJSM'] >= 1) & (x[ver + 'XRJSL'] >= 1), 'label'] = 1
    return x


def JSN_progression(x):
    x.loc[:, 'label'] = 0
    x.loc[(x['V01' + 'XRJSM'] - x['V00' + 'XRJSM']) > 0, 'label'] = 1
    x.loc[(x['V01' + 'XRJSL'] - x['V00' + 'XRJSL']) > 0, 'label'] = 1
    return x


def JSN_ever_progress(df):
    label = (sum([df[x] > df['V00XRJSM'] for x in oai_subjects.dict_variables['JSM'][1:3]])) + \
            (sum([df[x] > df['V00XRJSL'] for x in oai_subjects.dict_variables['JSL'][1:3]]))
    label = (label > 0)
    df['label'] = label.astype(np.int64)
    return df


def OAI_labels_process(df, scenerio):
    if scenerio == 'JSN00':
        return JSN_one_hot(df, 'V00')
    if scenerio == 'JSN00':
        return JSN_one_hot(df, 'V00')

    if 0:
        # JSN00
        subjects_train, subjects_val = \
            JSN_one_hot(subjects_train, 'V00'), JSN_one_hot(subjects_val, 'V00')

        # JSN 00-01 progress
        subjects_train, subjects_val = JSN_progression(subjects_train), JSN_progression(subjects_val)

        # JSN Ever Progress
        subjects_train, subjects_val = JSN_ever_progress(subjects_train), JSN_ever_progress(subjects_val)

