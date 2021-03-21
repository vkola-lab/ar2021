from loaders.loader_imorphics import *

args_d = {'mask_name': 'bone_resize_B_crop_all',
          'data_path': os.getenv("HOME") + '/Dataset/OAI_DESS_segmentation/',
          'scale': 0.5,
          'mask_used': [[1], [2, 3], [5, 6]],
          'interval': 4,
          'copy_channel': True,
          'pick': False,
          'method': 'automatic',
          'subject_splitting': '00',
          'val_percent': 0,
          'use_v01': False}

""" split range"""
eval_range = list(range(1, 10)) + list(range(71, 89))
train_range = list(range(10, 71))

""" data loader"""
t00 = LoaderImorphics(args_d, subjects_list=train_range, transform=None)
t01 = LoaderImorphics(args_d, subjects_list=list(range(10 + 88, 71 + 88)), transform=None)

train_loader = DataLoader(t00, batch_size=16, shuffle=True, num_workers=16, drop_last=False)
eval_set = LoaderImorphics(args_d, subjects_list=eval_range, transform=None)
eval_loader = DataLoader(eval_set, batch_size=16, shuffle=True, num_workers=16, drop_last=False)


t00_01 = LoaderImorphicsDual(args_d, subjects_list=(train_range, list(range(10 + 88, 71 + 88))), transform=None)

fc = []
for i in range(len(t00_01)):
    x = t00_01.__getitem__(i)
    mask00 = x[2][0][0, 0, ::]
    mask01 = x[2][0][1, 0, ::]
    fc00 = (mask00 == 1).sum()
    fc01 = (mask01 == 1).sum()
    fc.append([fc00, fc01])