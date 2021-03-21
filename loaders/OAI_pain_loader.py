import numpy as np
from collections import Counter
import glob, os, time
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def load_OAI_var():
    all_path = os.path.join(os.path.expanduser('~'), 'Dropbox')+ '/Z_DL/OAI_Labels/'
    print(all_path)
    all_var = glob.glob(all_path + '*.npy')
    all_var.sort()
    v = dict()
    for var in all_var:
        name = var.split('/')[-1].split('.')[0]
        v[name] = np.load(var, allow_pickle=True)

    return v

def get_OAI_pain_labels():
    Labels = dict()
    v = load_OAI_var()

    FreL_uni = v['fre_pain_l'][np.searchsorted(v['ID_main'], v['ID_uni_fre_pain'])]
    FreR_uni = v['fre_pain_r'][np.searchsorted(v['ID_main'], v['ID_uni_fre_pain'])]

    quality = np.logical_and(v['fail_uni_l'] == 0, v['fail_uni_r'] == 0)

    select_condition = 'np.logical_and(quality == 1, abs(v["WOMP_uni_l"]-v["WOMP_uni_r"]) >= 3)'
    pick = eval(select_condition)

    Labels['label'] = FreR_uni[pick]
    Labels['ID_selected'] = v['ID_uni_fre_pain'][pick]

    return Labels['label']


class OAIUnilateralPain(pl.LightningDataModule):
    def __init__(self, mode):
        self.mri_left = np.load('/media/ghc/GHc_data1/OAI_uni_pain/'
                                'unilateral_pain_left_womac3.npy')
        self.mri_right = np.load('/media/ghc/GHc_data1/OAI_uni_pain/'
                                'unilateral_pain_right_womac3.npy')
        self.labels = get_OAI_pain_labels()
        if mode == 'train':
            self.index = range(213, 710)
            #self.index = range(497)  # this split lead to lower accuracy
        elif mode == 'eval':
            self.index = range(213)
            #self.index = range(497, 710)

    def __len__(self):
        return self.labels[self.index].shape[0]

    def __getitem__(self, idx):
        index = self.index[idx]
        label = self.labels[index]

        l = self.mri_left[index, ::]
        r = self.mri_right[index, ::]

        # copy channel
        l = np.concatenate([l] * 3, 0)
        r = np.concatenate([r] * 3, 0)
        return idx, (l, r), (label, )


if __name__ == "__main__":
    args = {'batch_size': 1}

    train_set = OAIUnilateralPain(mode='train')
    val_set = OAIUnilateralPain(mode='eval')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=1, drop_last=False)
    eval_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=1, drop_last=False)
    x = train_set.__getitem__(10)

    for i, x in enumerate(eval_loader):
        print('  ')
