import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn
import matplotlib
import seaborn as sns
import matplotlib.colors as mc
import colorsys
from scipy.stats import ttest_ind

print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('scipy: {}'.format(scipy.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))

###############
# Preparation
###############

cli = pd.read_csv('df_extracted/merge1.csv')
sbl = pd.read_csv('df_extracted/SBL_0904.csv')  # Lateral 0-200 Medial
sbl_col_names = ['F' + str(i) for i in range(200)] + ['T' + str(i) for i in range(200)]

###############
# normalize SBL
###############
sbl_values = sbl.loc[:, sbl_col_names].values
for row in range(sbl_values.shape[0]):
    sbl_values[row, :] = sbl_values[row, :] / sbl_values[row, :].mean()  # normalize by the averaged val. of SBL
# flip left to right so left is medial side and right is lateral side
sbl_values = np.concatenate([np.fliplr(sbl_values[:, :200]), np.fliplr(sbl_values[:, 200:])], 1) # Medial 0-200 Lateral
sbl.loc[:, sbl_col_names] = sbl_values


#############
# sbl boxplot
#############
sbl_femur = pd.DataFrame(columns=['loc', 'sbl', 'V00XRJSM', 'V00XRJSL'])
sbl_tibia = pd.DataFrame(columns=['loc', 'sbl', 'V00XRJSM', 'V00XRJSL'])
for i in range(200):
    sbl_femur = sbl_femur.append(pd.DataFrame({'loc': i, 'sbl': sbl.loc[:, 'F' + str(i)].values,
                                               'V00XRJSM': cli['V00XRJSM'], 'V00XRJSL': cli['V00XRJSL']}))
    sbl_tibia = sbl_tibia.append(pd.DataFrame({'loc': i, 'sbl': sbl.loc[:, 'T' + str(i)].values,
                                               'V00XRJSM': cli['V00XRJSM'], 'V00XRJSL': cli['V00XRJSL']}))


def sbl_boxplot(data, text, baseline):
    flierprops = dict(markerfacecolor='0', markersize=0.3, linestyle='none')
    ax = sns.boxplot(x=data['loc'], y=data['sbl'], flierprops=flierprops, showmeans=False, width=0.6, linewidth=0.6)
    ax.xaxis.set_tick_params(width=1.75)
    ax.yaxis.set_tick_params(width=1.75)
    for _,s in ax.spines.items():
        s.set_linewidth(1.75)
    for i, artist in enumerate(ax.artists):
        control = baseline.loc[baseline['loc'] == i, 'sbl'].values
        compare = data.loc[data['loc'] == i, 'sbl'].values
        tstat, pval = ttest_ind(control, compare)
        if pval < 0.0001:
            col = (0.35, 0.70, 0.90)
        elif pval < 0.001:
            col = (0, 0.60, 0.50)
        elif pval < 0.01:
            col = (0.95, 0.90, 0.25)
        elif pval <= 0.05:
            col = (0.80, 0.4, 0)
        else:
            col = (150/255, 150/255, 150/255)
        artist.set_edgecolor(col)
        artist.set_facecolor('None')

        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            #line.set_mfc(col)
            #line.set_mec(col)

    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.set_xticklabels([0, ' ', 50, ' ', 100, ' ', 150, ' ', 200], fontname="arial", fontsize=16, weight = 'bold')
    ax.set_xlabel(" ", fontsize=1)
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticklabels([0, ' ', 1, ' ', 2, ' ', 3], fontname="arial", fontsize=16, weight = 'bold')
    ax.set_ylabel(" ", fontsize=1)
    plt.title(text, fontsize=20)
    plt.suptitle("")
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #ax.text(0.05, 0.95, text + ' N=' + str(data.shape[0] // 200), transform=ax.transAxes, fontsize=14,
           #verticalalignment='top', bbox=props)
    fig = ax.get_figure()
    fig.savefig('figures/sbl_box/'+ text+'.png', dpi=300)
    #plt.show()
    #plt.close()


baseline_femur = (sbl_femur.loc[(sbl_femur['V00XRJSM'] == 0) & (sbl_femur['V00XRJSL'] == 0), ['loc', 'sbl']])
baseline_tibia = (sbl_tibia.loc[(sbl_tibia['V00XRJSM'] == 0) & (sbl_tibia['V00XRJSL'] == 0), ['loc', 'sbl']])

for jsm in range(0, 4):
    sbl_boxplot(data=(sbl_femur.loc[(sbl_femur['V00XRJSM'] == jsm) & (sbl_femur['V00XRJSL'] == 0), ['loc', 'sbl']]),
                text='Femur, JSN Medial=' + str(jsm), baseline=baseline_femur)
    sbl_boxplot(data=(sbl_tibia.loc[(sbl_tibia['V00XRJSM'] == jsm) & (sbl_tibia['V00XRJSL'] == 0), ['loc', 'sbl']]),
                text='Tibia, JSN Medial=' + str(jsm), baseline=baseline_tibia)

for jsl in range(0, 4):
    sbl_boxplot(data=(sbl_femur.loc[(sbl_femur['V00XRJSM'] == 0) & (sbl_femur['V00XRJSL'] == jsl), ['loc', 'sbl']]),
                text='Femur, JSN Lateral=' + str(jsl), baseline=baseline_femur)
    sbl_boxplot(data=(sbl_tibia.loc[(sbl_tibia['V00XRJSM'] == 0) & (sbl_tibia['V00XRJSL'] == jsl), ['loc', 'sbl']]),
                text='Tibia, JSN Lateral=' + str(jsl), baseline=baseline_tibia)

################
# sbl difference
################
import matplotlib.font_manager as font_manager
def plot_sbl_difference(sbl, text, condition, condition_values, baseline):
    df = []
    for i in condition_values:
        df.append(np.expand_dims(sbl.loc[condition == i, sbl_col_names].values.mean(0) - baseline, 1))
    df = pd.DataFrame(np.concatenate(df, 1), columns=['Grade '+str(i) for i in condition_values])
    linestyle=['-', '--', '-.', ':']
    font = font_manager.FontProperties(family='arial', weight='bold', size=12)
    ax1 = df.iloc[:200, :]
    ax1 = ax1.plot(title='Femur SBL', style=linestyle, linewidth=1.5, fontsize=12)
    ax1.xaxis.set_tick_params(width=1.75)
    ax1.yaxis.set_tick_params(width=1.75)
    for _,s in ax1.spines.items():
        s.set_linewidth(1.75)
    ax1.set_xlabel('Frontal Axis')
    ax1.set_ylabel('SBL Average Difference')
    ax1.set_ylim(-0.4, 0.6)
    ax1.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax1.set_yticklabels([-0.4, ' ', -0.2, ' ', 0, ' ', 0.2, ' ', 0.4, ' ', 0.6], fontname="arial", fontsize=16, weight = 'bold')
    ax1.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax1.set_xticklabels([0, ' ', 50, ' ', 100, ' ', 150, ' ', 200], fontname="arial", fontsize=16, weight = 'bold')
    ax1.legend(prop=font, framealpha=5, labelspacing=0.3)
    fig = ax1.get_figure()
    fig.savefig('sbl_difference_'+text+'_femur.png', dpi=300)
    #plt.close()
    #plt.show()
    ax2 = df.iloc[200:, :]
    ax2 = ax2.plot(title='Tibia SBL', style=linestyle, linewidth=1.5, fontsize=12)
    ax2.xaxis.set_tick_params(width=1.75)
    ax2.yaxis.set_tick_params(width=1.75)
    for _,s in ax2.spines.items():
        s.set_linewidth(1.75)
    ax2.set_xlabel('Frontal Axis')
    ax2.set_ylabel('SBL Average Difference')
    ax2.set_ylim(-0.4, 0.6)
    ax2.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax2.set_yticklabels([-0.4, ' ', -0.2, ' ', 0, ' ', 0.2, ' ', 0.4, ' ', 0.6], fontname="arial", fontsize=16, weight = 'bold')
    ax2.set_xticks([200, 225, 250, 275, 300, 325, 350, 375, 400])
    ax2.set_xticklabels([0, ' ', 50, ' ', 100, ' ', 150, ' ', 200], fontname="arial", fontsize=16, weight = 'bold')
    ax2.legend(prop=font, framealpha=5, labelspacing=0.3)
    fig = ax2.get_figure()
    fig.savefig('sbl_difference_'+text+'_tibia.png', dpi=300)
    #plt.close()
    #plt.show()

#  define baseline
sbl_jsn_0_mean = sbl.loc[(cli['V00XRJSM'] == 0) & (cli['V00XRJSL'] == 0), sbl_col_names].values.mean(0)
#sbl_KL_0_mean = sbl.loc[cli['V00XRKL'] == 0, sbl_col_names].values.mean(0)
#sbl_pain_0_mean = sbl.loc[cli['V00WOMKP#'] == 0, sbl_col_names].values.mean(0)
baseline = sbl_jsn_0_mean


# plot the difference in sbl compared to baselines
plot_sbl_difference(sbl, text='KL', condition=cli['V00XRKL'],
                    condition_values=[0, 1, 2, 3, 4], baseline=baseline)
plot_sbl_difference(sbl, text='JSN Medial', condition=cli['V00XRJSM'],
                    condition_values=[0, 1, 2, 3], baseline=baseline)
plot_sbl_difference(sbl, text='JSN Lateral', condition=cli['V00XRJSL'],
                    condition_values=[0, 1, 2, 3], baseline=baseline)

#####################
# Calculate Odd Ratio
#####################
def get_OR(x, outcome, condition):
    #subjects w/ condition. SBLs and outcomes.
    x = x.loc[condition]
    outcome = outcome.loc[condition]
    print("total number of subject w/ condition:" + str(outcome.shape[0]))

    #find SBL quantile cuttoffs of subjects w/ condition
    quantile = [np.quantile(x, i) for i in [0, 0.25, 0.5, 0.75, 1]]
    print("_______QUANTILE CUTOFFS_______")
    print(quantile)
    OR = np.zeros((2, 2))
    i = 0

    #find outcome data for subjects in 1st quantile
    found = outcome[(x > quantile[i]) & (x <= quantile[i + 1])]
    print("total number of subjects in 1st quantile" + str(found.shape[0]))
    OR[1, 0] = (found == 1).sum()
    OR[1, 1] = (found == 0).sum()

    #mean SBL difference abs in 1st quantile
    test = x[(x > quantile[i]) & (x <= quantile[i + 1])]
    print("mean, std, median of 1st quantile")
    print(test.mean())
    print(test.std())
    print(test.median())
    print("total number of subjects in 1st quantile" + str(test.shape[0]))

    to_print=''
    #find outcome data for subjects in 2nd/3rd/4th quantile then compare w/ 1st quantile.
    for i in range(1, 4):
        found = outcome[(x > quantile[i]) & (x <= quantile[i + 1])]
        q = i + 1
        print("total number of subjects in quantile" + str(q) + ": " + str(found.shape[0]))
        OR[0, 0] = (found == 1).sum()
        OR[0, 1] = (found == 0).sum()

        #calculate oddsratio
        oddsratio, pvalue = stats.fisher_exact(OR)
        LOR = np.log(OR[0,0]) + np.log(OR[1,1]) - np.log(OR[0,1]) - np.log(OR[1,0])
        SE = np.sqrt(np.sum(1/OR.astype(np.float64)))
        LCL = np.exp(LOR - 1.96*SE)
        UCL = np.exp(LOR + 1.96*SE)
        if pvalue <= 0.05:
            significance = '*'
        else:
            significance = ' '
        #print("Quantile: ", i + 1, "OR: ", oddsratio, "p-Value: %.4f"+significance % pvalue)
        to_print = to_print+("Q%d, OR: %.2f, p: %.4f"+significance+" CI: %.2f, %.2f") % (i + 1, oddsratio, pvalue, LCL, UCL) + '  '
    print("_______OUTCOMES TABLE for CALC ODDS RATIO________")
    print("1st column = outcome true. 1st row = 1st quantile. 2nd row = 2nd/3rd/4th quantile")
    print(OR)
    print("_______ODDS RATIO________")
    print(to_print)
    
# sum of all the absolute value of sbl difference.
baseline = sbl_jsn_0_mean
sbl_difference = (sbl.loc[:, sbl_col_names].sub(baseline, axis=1))
sbl_difference_absolute = sbl_difference.abs().sum(1)

#get OR
condition = (cli['V00XRKL'] >= 3)
conditions = {'KL≥3': cli['V00XRKL'] >= 3,
              'JSN Medial≥1': cli['V00XRJSM'] >= 1,
              'JSN Lateral≥1': cli['V00XRJSL'] >= 1,}

outcomes = {'Moderate Pain:': ((cli['V00WOMKP#'] >= 4) & (cli['V00WOMKP#'] < 8)),
            'Severe Pain:': (cli['V00WOMKP#'] >= 8),
            'Moderate Disability:': ((cli['V00WOMADL#'] >= 20) & (cli['V00WOMADL#'] < 35)),
            'Severe Disability:': (cli['V00WOMADL#'] >= 35),
            'Future TKR': (cli['V99E#KRPSN'] >= 1)}

for outcome in outcomes.keys():
    for condition in conditions.keys():
        print(outcome + ' | ' + condition)
        get_OR(x=sbl_difference_absolute, outcome=outcomes[outcome],
               condition=conditions[condition])
    print('')


#########################
#QUARTILE MEAN/SD/MEDIAN
#########################

def get_stats(x, outcome, condition):
    x = x.loc[condition]
    outcome = outcome.loc[condition]
    df = pd.DataFrame({'sbl_difference': x, 'outcome': outcome})
    df.loc[(df['outcome'] == True), 'Q'] = pd.qcut(df['sbl_difference'], 4, labels=('Q1', 'Q2', 'Q3', 'Q4'))
    df['Q'] = df.Q.astype(str)

    groups = df['Q']   
    q1 = (groups == 'Q1')
    q2 = (groups == 'Q2')  
    q3 = (groups == 'Q3')
    q4 = (groups == 'Q4')

    q1.name = 'Quartile 1'
    q2.name = 'Quartile 2'
    q3.name = 'Quartile 3'
    q4.name = 'Quartile 4'

    stat = df['sbl_difference']
    quartiles = [q1, q2, q3, q4]
    for quartile in quartiles:
        print("total subjects in " + quartile.name + ": " + str((quartile==1).sum()))
        qmn = stat[quartile].mean()
        qsd = stat[quartile].std()
        qmd = stat[quartile].median()
        print(quartile.name+' Mean: ', qmn, ' SD: ', qsd, ' Median: ', qmd)

conditions = {'JSN Medial≥1': cli['V00XRJSM'] >= 1,
            'JSN Lateral≥1': cli['V00XRJSL'] >= 1}

outcomes = {'Moderate Pain:': ((cli['V00WOMKP#'] >= 4) & (cli['V00WOMKP#'] < 8)),
            'Severe Pain:': (cli['V00WOMKP#'] >= 8),
            'Moderate Disability:': ((cli['V00WOMADL#'] >= 20) & (cli['V00WOMADL#'] < 35)),
            'Severe Disability:': (cli['V00WOMADL#'] >= 35),
            'Future TKR': (cli['V99E#KRPSN'] >= 1)}

for outcome in outcomes.keys():
    for condition in conditions.keys():
        print(outcome + ' | ' + condition)
        get_stats(x=sbl_difference_absolute, outcome=outcomes[outcome],
               condition=conditions[condition])
    print('')
