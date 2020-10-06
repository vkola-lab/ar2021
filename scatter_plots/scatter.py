import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

with open('subchondral_length.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    patient_id = []
    slice_normalized = []
    KL = []
    femur_truth = []
    femur_pred = []
    tibia_truth = []
    tibia_pred = []
    colors = []

    for row in spamreader:
        row = row[0]
        row = row.split(',')
        patient_id.append(row[0])
        slice_normalized.append(float(row[1]))
        KL.append(row[2])
        femur_truth.append(float(row[3]))
        femur_pred.append(float(row[4]))
        tibia_truth.append(float(row[5]))
        tibia_pred.append(float(row[6]))

    print slice_normalized
    N = 794
    area = np.pi*8

    plt.figure(figsize=(7,5))
    plot1 = plt.scatter(slice_normalized, femur_truth, s=area, c=('#1f77b4'), marker = 'o', alpha = 0.5)
    plot2 = plt.scatter(slice_normalized, femur_pred, s=area, c=('#9467bd'), marker = '+', alpha=0.5)
    plt.title('Femur truth vs Femur pred')
    plt.xlabel('Slice Normalized')
    plt.ylabel('Subchondral Length (pixels)')
    plt.legend((plot1, plot2), ('Truth', 'Pred'))
    plt.text(0.1,0, 'p=0.1385')
    plt.show()

    plt.figure(figsize=(7,5))
    plot1 = plt.scatter(slice_normalized, tibia_truth, s=area, c=('#1f77b4'), marker = 'o', alpha = 0.5)
    plot2 = plt.scatter(slice_normalized, tibia_pred, s=area, c=('#9467bd'), marker = '+', alpha=0.5)
    plt.title('Tibia truth vs Tibia pred')
    plt.xlabel('Slice Normalized')
    plt.ylabel('Subchondral Length (pixels)')
    plt.legend((plot1, plot2), ('Truth', 'Pred'))
    plt.text(0.1, 0, 'p=0.6833')
    plt.show()

    plt.figure(figsize=(7,5))
    plot1 = plt.scatter(slice_normalized[:31], femur_truth[:31], s=area, c='#1f77b4', marker = 'o', alpha = 0.5)
    plot2 = plt.scatter(slice_normalized[31:300], femur_truth[31:300], s=area, c=('#9467bd'), marker = '+', alpha=0.5)
    plot3 = plt.scatter(slice_normalized[300:765], femur_truth[300:765], s=area, c=('#2ca02c'), marker = 'x', alpha = 0.5)
    plot4 = plt.scatter(slice_normalized[765:], femur_truth[765:], s=area, c=('#ff7f0e'), marker = 's', alpha=0.5)
    plt.title('Femur Truth KL Comparison')
    plt.xlabel('Slice Normalized')
    plt.ylabel('Subchondral Length (pixels)')
    plt.legend((plot1, plot2, plot3, plot4), ('KL1', 'KL2', 'KL3', 'KL4'))
    plt.text(0.1, 0, 'p=0.185')
    plt.show()

    plt.figure(figsize=(7,5))
    plot1 = plt.scatter(slice_normalized[:31], femur_pred[:31], s=area, c='#1f77b4', marker = 'o', alpha = 0.5)
    plot2 = plt.scatter(slice_normalized[31:300], femur_pred[31:300], s=area, c=('#9467bd'), marker = '+', alpha=0.5)
    plot3 = plt.scatter(slice_normalized[300:765], femur_pred[300:765], s=area, c=('#2ca02c'), marker = 'x', alpha = 0.5)
    plot4 = plt.scatter(slice_normalized[765:], femur_pred[765:], s=area, c=('#ff7f0e'), marker = 's', alpha=0.5)
    plt.title('Femur Predicted KL Comparison')
    plt.xlabel('Slice Normalized')
    plt.ylabel('Subchondral Length (pixels)')
    plt.legend((plot1, plot2, plot3, plot4), ('KL1', 'KL2', 'KL3', 'KL4'))
    plt.text(0.1, 0, 'p=0.434')
    plt.show()

    plt.figure(figsize=(7,5))
    plot1 = plt.scatter(slice_normalized[:31], tibia_truth[:31], s=area, c='#1f77b4', marker = 'o', alpha = 0.5)
    plot2 = plt.scatter(slice_normalized[31:300], tibia_truth[31:300], s=area, c=('#9467bd'), marker = '+', alpha=0.5)
    plot3 = plt.scatter(slice_normalized[300:765], tibia_truth[300:765], s=area, c=('#2ca02c'), marker = 'x', alpha = 0.5)
    plot4 = plt.scatter(slice_normalized[765:], tibia_truth[765:], s=area, c=('#ff7f0e'), marker = 's', alpha=0.5)
    plt.title('Tibia Truth KL Comparison')
    plt.xlabel('Slice Normalized')
    plt.ylabel('Subchondral Length (pixels)')
    plt.legend((plot1, plot2, plot3, plot4), ('KL1', 'KL2', 'KL3', 'KL4'))
    plt.text(0.1, 0, 'p=0.259')
    plt.show()

    plt.figure(figsize=(7,5))
    plot1 = plt.scatter(slice_normalized[:31], tibia_pred[:31], s=area, c='#1f77b4', marker = 'o', alpha = 0.5)
    plot2 = plt.scatter(slice_normalized[31:300], tibia_pred[31:300], s=area, c=('#9467bd'), marker = '+', alpha=0.5)
    plot3 = plt.scatter(slice_normalized[300:765], tibia_pred[300:765], s=area, c=('#2ca02c'), marker = 'x', alpha = 0.5)
    plot4 = plt.scatter(slice_normalized[765:], tibia_pred[765:], s=area, c=('#ff7f0e'), marker = 's', alpha=0.5)
    plt.title('Tibia Predicted KL Comparison')
    plt.xlabel('Slice Normalized')
    plt.ylabel('Subchondral Length (pixels)')
    plt.legend((plot1, plot2, plot3, plot4), ('KL1', 'KL2', 'KL3', 'KL4'))
    plt.text(0.1, 0, 'p=0.699')
    plt.show()





