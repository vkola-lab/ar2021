# Subchondral bone length in knee osteoarthritis: A deep learning driven imaging measure and its association with radiographic and clinical outcomes

This work is published in _Arthritis & Rheumatology_ (https://doi.org/10.1002/art.41808).

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
        ├── testing              # DESS MRI for predictions
    ├── dess_utils               # Tools to create segmentations and to perform statistical analysis
        ├── SBL_statistics.py    # Statistical analysis of SBL data
    ├── engine                   # Engine of Pytorch Lightning
    ├── loaders                  # Loaders for knee DESS MRI images and annotations
    ├── logs                     # Training logs of Pytorch
    ├── models                   # Definition of Pytorch models
    ├── ln_segmentation.py       # main script to perform model traning of bone and cartilage segmentation
    ├── prediction.py            # calculate SBL based on predicted bone and cartilages masks
    ├── postprocess.py           # cleaning segmentation results, create SBL based on predicted bone and cartilages masks
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

#### Neural Network Training of Automatic Segmentation
```bash
python ln_segmentation.py
```
#### Sample Scripts for prediction of bone and cartilage segmentation
```bash
python prediction.py
```

#### Sample Scripts for Postprocessing
```bash
python postprocess.py
```

## Stastistics
#### Demographics
#### Subchondral Bone Length Measurement
#### Odds Ratio
#### Box plots
#### SBL Difference Plot

## Others
#### Run script
```bash
cd ~/boneshape/sbl_OR/

python3 sbl_boxplot.py
```
#### Prerequisites
SBL data in ~/boneshape/df_extracted/SBL_0904.csv. Each row corresponds to a subject knee.
merge1.csv in ~/boneshape/df_extracted/merge1.csv contains the clinical data. Each row corresponds to a patient, while each column corresponds to demographic/clinical info. Index order must be maintained in merge1.csv and SBL_0904.csv.

Dependencies include:
1. pandas version 1.1.5
2. numpy version 1.19.5
3. scipy version 1.5.4
4. matplotlib version 3.3.4
5. seaborn version 0.11.1
6. sklearn version 0.24.1
7. colorsys

#### Results
sbl_boxplot.py script will produce the odds ratio data, box plots, and, SBL Difference Plot. Generated figures are located in ~/boneshape/figures.
