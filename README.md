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


