# Boneshape

This is the repository of the 2021 paper "Population-level measurement of cartilage and bone in knee osteoarthritis and their association with imaging markers using deep learning"

## Segmentation of MRI-Based Knee Shape

## Knee Shape Segmentation and Classification
#### Prerequisites

The tool was developed based on the following dependencies:

1. PyTorch (1.1 or later).
2. PyTorch Lightning (1.7 or later).
3. NumPy (1.16 or later).
4. Scipy (1.30 or later)
5. scikit-learn (0.21.2 or later)

#### Data files
Pairs of images and masks of segmentation should be put in:
```bash
data/CASE_NAME/train_imgs/*.png
data/CASE_NAME/train_masks/*.png
```
for training dataset and
```bash
data/CASE_NAME/eval_imgs/*.png
data/CASE_NAME/eval_masks/*.png
```
for validation dataset.
The images for testing the segmentation results should be put in:
```bash
data/CASE_NAME/test_imgs/*.png
```
And the segmentation results will be created in:
```bash
data/CASE_NAME/test_masks/*.png
```

#### Preprocessing
```bash
python ln_main.py -t pre -d eval
```
#### Neural Network Training of Segmentation
```bash
python ln_main.py -t seg -d train
```
#### Neural Network Interference of Segmentation
```bash
python ln_main.py -t seg -d eval
```

## Stastistics
#### Demographics
#### Subchondral Bone Length Measurement


## Others
#### Run script
```bash
cd ~/boneshape/f_and_t_tests/

python f_and_t_tests.py
```
#### Results
Results of T value, P value, and P value < 0.05 significance will be in files:
tibia_lateral.csv (tibia with no JSN vs tibia with JSN in lateral)
femur_lateral.csv (femur with no JSN vs femur with JSN in lateral)
femur_medial.csv (femur with no JSN vs femur with JSN in medial)
tibia_medial.csv (tibia with no JSN vs tibia with JSN in medial)

Each row is a slice.

Script will also plot the T test results as a scatter plot.
Script will also plot the difference of means for each slice, color coded by the T test p values.

## F and T tests
#### Prerequisites
Instructions to run F and T tests on dataset.
Dataset in form of npy files. In directory f_and_t_tests:

JSL_left.npy, JSL_right.npy, JSM_left.npy, JSM_right.npy contain data about grade of joint space narrowing (JSN) in the lateral (JSL) or medial (JSM) side of knee.

LF_all.npy, RT_all.npy, LT_all.npy, RF_all.npy contain data about the subchondral bone length in each MRI slice of tibia (RT, LT) and femur (RF, LF) of knee.

npy files must be in the same directory as the f_and_t_tests.py file.
