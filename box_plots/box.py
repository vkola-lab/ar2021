import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 14})

#slice_normalized = pd.Series(list(range(100)), index=index)

KL_array = np.load('KL00_R.npy')
KL = pd.DataFrame(KL_array, columns = ['id', 'grade'])

tibia_array = np.load('tibia_cartilage.npy')
tibia = pd.DataFrame(tibia_array)

femur_array = np.load('femur_cartilage.npy')
femur = pd.DataFrame(femur_array)


plt.figure(); femur.boxplot()
plt.title('Femur Subchondral Length Distribution')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()


plt.figure(); tibia.boxplot()
plt.title('Tibia Subchondral Length Distribution')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()



KL_0 = KL.loc[KL.grade == 0.0]
KL_1 = KL.loc[KL.grade == 1.0]
KL_2 = KL.loc[KL.grade == 2.0]
KL_3 = KL.loc[KL.grade == 3.0]
KL_4 = KL.loc[KL.grade == 4.0]

femur_0 = femur.loc[KL_0.index.values]
femur_1 = femur.loc[KL_1.index.values]
femur_2 = femur.loc[KL_2.index.values]
femur_3 = femur.loc[KL_3.index.values]
femur_4 = femur.loc[KL_4.index.values]

tibia_0 = tibia.loc[KL_0.index.values]
tibia_1 = tibia.loc[KL_1.index.values]
tibia_2 = tibia.loc[KL_2.index.values]
tibia_3 = tibia.loc[KL_3.index.values]
tibia_4 = tibia.loc[KL_4.index.values]


plt.figure(); femur_0.boxplot()
plt.title('Femur Subchondral Length Distribution, KL Grade 0')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); femur_1.boxplot()
plt.title('Femur Subchondral Length Distribution, KL Grade 1')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); femur_2.boxplot()
plt.title('Femur Subchondral Length Distribution, KL Grade 2')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); femur_3.boxplot()
plt.title('Femur Subchondral Length Distribution, KL Grade 3')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); femur_4.boxplot()
plt.title('Femur Subchondral Length Distribution, KL Grade 4')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()


plt.figure(); tibia_0.boxplot()
plt.title('Tibia Subchondral Length Distribution, KL Grade 0')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); tibia_1.boxplot()
plt.title('Tibia Subchondral Length Distribution, KL Grade 1')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); tibia_2.boxplot()
plt.title('Tibia Subchondral Length Distribution, KL Grade 2')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); tibia_3.boxplot()
plt.title('Tibia Subchondral Length Distribution, KL Grade 3')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()

plt.figure(); tibia_4.boxplot()
plt.title('Tibia Subchondral Length Distribution, KL Grade 4')
plt.xlabel('Slice Normalized')
plt.ylabel('Subchondral Length Normalized')
plt.xticks(np.arange(0, 101, 25), ('0', '0.25', '0.5', '0.75', '1'))
plt.show()
