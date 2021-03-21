import pandas as pd
import numpy as np
import glob, os


class OAISubejctsLabels():
    def __init__(self, source, tags_used):
        self.source = source
        self.tags_used = tags_used
        self.xr = pd.read_csv(os.path.expanduser("~") + '/Dropbox/Z_DL/scripts/OAI_stats/collected/XR.csv')

        self.dict_variables = dict()
        self.dict_variables['KL'] = ['V' + str(x).zfill(2) + 'XRKL' for x in [0, 1, 3, 5, 6, 8, 10]]
        self.dict_variables['JSM'] = ['V' + str(x).zfill(2) + 'XRJSM' for x in [0, 1, 3, 5, 6, 8, 10]]
        self.dict_variables['JSL'] = ['V' + str(x).zfill(2) + 'XRJSL' for x in [0, 1, 3, 5, 6, 8, 10]]

    def read_oai_XR(self, df):
        """ Read OAI XR reading labels"""
        # STUPID df is XR reading with merged projects
        xr = self.xr
        xr = xr.rename(columns={"ID": 'subjects', "SIDE": "sides"})
        #xr['sides'] = [(s, ) for s in xr['sides']]
        df = pd.merge(df, xr, how='inner', on=['subjects', 'sides'])
        return df

    def find_subjects_from_folders(self, tags_used):
        """
        scan all the folders in tags_used and find the subjects that exist in all the folders
        """
        found_ids = dict()
        for i, tag in enumerate(tags_used):
            if 'RIGHT' in tag:
                side = 1
            elif 'LEFT' in tag:
                side = 2
            temp = [(int(x.split('/')[-1].split('.')[0]), side, tag) for x in
                    glob.glob(self.source + tag + '/*')]
            temp.sort()
            found_ids[i] = pd.DataFrame(temp, columns=['subjects', 'sides', 'tag'])
        return found_ids

    def subjects_concat(self):
        found_ids = self.find_subjects_from_folders(self.tags_used)
        return pd.concat(found_ids)

    def find_subjects_from_moaks(self):
        # STUPID
        moaks00 = pd.read_csv(os.path.expanduser("~") + '/Dropbox/Z_DL/scripts/OAI_stats/collected/moaks00.csv')
        return pd.DataFrame({'subjects': moaks00['ID'], 'sides': moaks00['SIDE']})

    def train_XR_validation_MOAKS(self):
        xr_subjects = self.subjects_concat().reset_index(drop=True).reset_index()
        moaks00 = self.find_subjects_from_moaks()

        validation = pd.merge(xr_subjects, moaks00, how='inner', on=['subjects', 'sides'])
        training = xr_subjects[~xr_subjects['index'].isin(validation['index'])]

        training = self.read_oai_XR(training)
        validation = self.read_oai_XR(validation)

        return training, validation


def merge_multiple_dfs(dfs, how='inner', on=['subjects']):
    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = pd.merge(merged, dfs[i], how=how, on=on)
    return merged


def merge_sides_tags(merged):
    cols = merged.columns
    all_side = [a for a in cols if 'side' in a]
    all_tag = [a for a in cols if 'tag' in a]
    merged['sides'] = list(zip(*[merged[x] for x in all_side]))
    merged['tags'] = list(zip(*[merged[x] for x in all_tag]))
    return merged[['subjects', 'sides', 'tags']]


if __name__ == "__main__":
    args_d = {'img_dir': '/media/ghc/GHc_data1/dess_segmentation_processed/',
              'tags_used': ['SAG_3D_DESS_LEFT_00_2D', 'SAG_3D_DESS_RIGHT_00_2D']}

    oai_subjects = OAISubejctsLabels(source=args_d['img_dir'],
                                     tags_used=args_d['tags_used'])

    subjects_train, subjects_val = oai_subjects.train_XR_validation_MOAKS()



