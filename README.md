# Boneshape

## F and T tests
#### Prerequisites
Instructions to run F and T tests on dataset.
Dataset in form of npy files. In directory f_and_t_tests:

JSL_left.npy, JSL_right.npy, JSM_left.npy, JSM_right.npy contain data about grade of joint space narrowing (JSN) in the lateral (JSL) or medial (JSM) side of knee.

LF_all.npy, RT_all.npy, LT_all.npy, RF_all.npy contain data about the subchondral bone length in each MRI slice of tibia (RT, LT) and femur (RF, LF) of knee.

npy files must be in the same directory as the f_and_t_tests.py file.

Python version 2.7.17 used.
numpy version 1.16.6 used.
pandas version 0.24.2 used.
matplotlib version 2.2.5 used.
scipy version 1.2.3 used.

Packages required:
```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scipy
```

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

## Knee Shape Segmentation and Classification
#### Prerequisites

The tool was developed based on the following dependencies:

1. PyTorch (1.1 or greater).
2. NumPy (1.16 or greater).
3. Scipy (1.30 or greater)
5. scikit-learn (0.21.2 or greater)

Please note that the dependencies require Python 3.6 or greater. We recommend installation and maintenance of all packages using [`conda`](https://www.anaconda.com/). For installation of GPU accelerated PyTorch, additional effort may be required. Please check the official websites of [PyTorch](https://pytorch.org/get-started/locally/) and [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions.

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
And the labels used for classification should be put as:
```bash
data/CASE_NAME/label.npy
```

## <a name="Arguments"></a>Arguments

`--case_name`:
>> name of the study.

`--epochs`: *int*
>> Number of epochs.

`-b`: *int*
>> Batch size

`--bu`: *int*
>> Batch size for accumulated gradient.

`--lr`: *float*
>> Learning rate, default is 1e-4

`--ini`:
>> name of the saved .ini file




