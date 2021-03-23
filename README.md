# Subchondral bone length in knee osteoarthritis: A deep learning driven imaging measure and its association with radiographic and clinical outcomes

## Prerequisites

The tool was developed based on the following dependencies:

1. PyTorch (1.1 or later).
2. PyTorch Lightning (1.3 or later).
3. NumPy (1.16 or later).
4. Scipy (1.30 or later)
5. scikit-learn (0.21.2 or later)
6. Tensorboard

## Project Structures

    ├── checkpoints              # Saved checkpoints of the U-Net models
    ├── data                     
        ├── dess_annotated       # Annotated DESS MRI images for training of the neural network
            ├── img                MRI images in .jpg format
                ├── 1_10.jpg
                ├── ...
            ├── masks
                ├── fc             Annotated masks of femur cartilage (fc), medial tibia cartilage (mtc), lateral tibia cartilage (ltc), etc...
                    ├── 1_10.jpg
                    ├── ...
                ├── mtc
                ├── ltc
                ├── ...
        ├── dess_mri             # DESS MRI for predictions
        ├── predicted            # Predicted bone and cartilages masks
    ├── dess_utils               # Tools to create segmentations and to perform statistical analysis
        ├── dess_prediction.py   # calculate SBL based on predicted bone and cartilages masks
        ├── dess_process.py      # cleaning segmentation results, create SBL based on predicted bone and cartilages masks
        ├── SBL_statistics.py    # Statistical analysis of SBL data
    ├── engine                   # Engine of Pytorch Lightning
    ├── loaders                  # Loaders for knee DESS MRI images and annotations
    ├── logs                     # Training logs of Pytorch
    ├── models                   # Definition of Pytorch models
    ├── ln_segmentation.py       # main script to perform model traning of bone and cartilage segmentation
    └── README.md

## Segmentation of MRI-Based Knee Shape

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
python ln_segmentation.py -t pre -d eval
```
#### Neural Network Training of Segmentation
```bash
python ln_segmentation.py -t seg -d train
```
#### Neural Network Interference of Segmentation
```bash
python ln_segmentation.py -t seg -d eval
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
