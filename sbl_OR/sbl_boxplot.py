import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
import seaborn as sns
import matplotlib.colors as mc
import colorsys
from scipy.stats import ttest_ind

###############
# Preparation
###############

cli = pd.read_csv('df_extracted/merge1.csv')
sbl = pd.read_csv('df_extracted/SBL_0904.csv')  # Lateral 0-200 Medial
sbl_col_names = ['F' + str(i) for i in range(200)] + ['T' + str(i) for i in range(200)]

imorphics = pd.read_csv('df_extracted/imorphics.csv')
imorphics = pd.merge(cli, imorphics, how='inner')

###############
# normalize SBL
###############
sbl_values = sbl.loc[:, sbl_col_names].values
for row in range(sbl_values.shape[0]):
    #sbl_values[row, :] = sbl_values[row, :] / sbl_values[row, :200].max()  # normalize by the max value in femur SBL
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
    ax = sns.boxplot(x=data['loc'], y=data['sbl'], showfliers=False, showmeans=False, color='w')
    for i, artist in enumerate(ax.artists):
        control = baseline.loc[baseline['loc'] == i, 'sbl'].values
        compare = data.loc[data['loc'] == i, 'sbl'].values
        tstat, pval = ttest_ind(control, compare)
        if pval < 0.0001:
            col = (55/255, 0, 0)
        elif pval < 0.001:
            col = (1, 0, 0)
        elif pval < 0.01:
            col = (1, 105/255, 0)
        elif pval <= 0.05:
            col = (1, 1, 0)
        else:
            col = (0, 1, 0)

        artist.set_edgecolor(col)

        if 1:
            for j in range(i * 5, i * 5 + 5):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)
                line.set_linewidth(0.5)  # ADDITIONAL ADJUSTMENT

    ax.set_xticks([0,  20])
    ax.set_xlabel("Frontal Axis", fontsize=20)
    ax.set_ylabel("Normalized SBL", fontsize=20)
    plt.title("Medial <-----|-----> Lateral", fontsize=20)
    plt.suptitle("")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text + ' N=' + str(data.shape[0] // 200), transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    fig = ax.get_figure()

    fig.savefig('figures/sbl_box/'+text+'.png')
    plt.close()


#baseline_femur = (sbl_femur.loc[(sbl_femur['V00XRJSM'] == 0), ['loc', 'sbl']])
#baseline_tibia = (sbl_femur.loc[(sbl_femur['V00XRJSM'] == 0), ['loc', 'sbl']])
baseline_femur = (sbl_femur.loc[(sbl_femur['V00XRJSM'] == 0) & (sbl_femur['V00XRJSL'] == 0), ['loc', 'sbl']])
baseline_tibia = (sbl_tibia.loc[(sbl_tibia['V00XRJSM'] == 0) & (sbl_tibia['V00XRJSL'] == 0), ['loc', 'sbl']])


for jsm in range(0, 4):
    sbl_boxplot(data=(sbl_femur.loc[(sbl_femur['V00XRJSM'] == jsm) & (sbl_femur['V00XRJSL'] == 0), ['loc', 'sbl']]),
                text='Femur, JSN Medial=' + str(jsm), baseline=baseline_femur)
    sbl_boxplot(data=(sbl_tibia.loc[(sbl_tibia['V00XRJSM'] == jsm) & (sbl_femur['V00XRJSL'] == 0), ['loc', 'sbl']]),
                text='Tibia, JSN Medial=' + str(jsm), baseline=baseline_tibia)

for jsl in range(0, 4):
    sbl_boxplot(data=(sbl_femur.loc[(sbl_femur['V00XRJSM'] == 0) & (sbl_femur['V00XRJSL'] == jsl), ['loc', 'sbl']]),
                text='Femur, JSN Lateral=' + str(jsl), baseline=baseline_femur)
    sbl_boxplot(data=(sbl_tibia.loc[(sbl_femur['V00XRJSM'] == 0) & (sbl_tibia['V00XRJSL'] == jsl), ['loc', 'sbl']]),
                text='Tibia, JSN Lateral=' + str(jsl), baseline=baseline_tibia)


################
# sbl difference
################
def plot_sbl_difference(sbl, text, condition, condition_values, baseline):
    df = []
    for i in condition_values:
        df.append(np.expand_dims(sbl.loc[condition == i, sbl_col_names].values.mean(0) - baseline, 1))
    df = pd.DataFrame(np.concatenate(df, 1), columns=[text+' '+str(i) for i in condition_values])
    ax1 = df.iloc[:200, :]
    ax1 = ax1.plot(title='Femur SBL')
    ax1.set_xlabel('Frontal Axis')
    ax1.set_ylabel('SBL Average Difference')
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xticks([0, 200])
    ax1.set_xticklabels([0, 1])
    fig = ax1.get_figure()
    fig.savefig('figures/sbl_difference/'+text+'_femur.png')
    plt.close()
    ax2 = df.iloc[200:, :]
    ax2 = ax2.plot(title='Tibia SBL')
    ax2.set_xlabel('Frontal Axis')
    ax2.set_ylabel('SBL Average Difference')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xticks([200, 400], [0, 1])
    ax2.set_xticklabels([0, 1])
    fig = ax2.get_figure()
    fig.savefig('figures/sbl_difference/'+text+'_tibia.png')
    plt.close()

#  define baseline
sbl_jsn_0_mean = sbl.loc[(cli['V00XRJSM'] == 0) & (cli['V00XRJSL'] == 0), sbl_col_names].values.mean(0)
sbl_KL_0_mean = sbl.loc[cli['V00XRKL'] == 0, sbl_col_names].values.mean(0)
sbl_pain_0_mean = sbl.loc[cli['V00WOMKP#'] == 0, sbl_col_names].values.mean(0)
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
    x = x.loc[condition]
    outcome = outcome.loc[condition]
    quantile = [np.quantile(x, i) for i in [0, 0.25, 0.5, 0.75, 1]]
    OR = np.zeros((2, 2))
    i = 0
    found = outcome[(x > quantile[i]) & (x <= quantile[i + 1])]
    OR[1, 0] = (found == 1).sum()
    OR[1, 1] = (found == 0).sum()

    to_print=''
    for i in range(1, 4):
        found = outcome[(x > quantile[i]) & (x <= quantile[i + 1])]
        OR[0, 0] = (found == 1).sum()
        OR[0, 1] = (found == 0).sum()
        oddsratio, pvalue = stats.fisher_exact(OR)
        if pvalue <= 0.05:
            significance = '*'
        else:
            significance = ' '
        #print("Quantile: ", i + 1, "OR: ", oddsratio, "p-Value: %.4f"+significance % pvalue)
        to_print = to_print+("Q%d, OR: %.4f, p: %.4f"+significance) % (i + 1, oddsratio, pvalue) + '  '
    print(to_print)

# sbl difference absolute
baseline = sbl_jsn_0_mean
sbl_difference = (sbl.loc[:, sbl_col_names].sub(baseline, axis=1))
sbl_difference_absolute = sbl_difference.abs().sum(1)

#############
# PCA
#############
pca = PCA(n_components=2)
JSM = sbl_difference.loc[(cli['V00XRJSM'] >= 1) & (cli['V00XRJSL'] == 0), :].values
JSL = sbl_difference.loc[(cli['V00XRJSM'] == 0) & (cli['V00XRJSL'] >= 1), :].values

pca = PCA(n_components=2)
pca.fit(sbl_difference.values)
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(pca.components_[0, :200][::-1])
matplotlib.pyplot.plot(pca.components_[0, 200:][::-1])

# PCA values
eigen = np.zeros((sbl_difference.shape[0]))
for i in range(sbl_difference.shape[0]):
    eigen[i] = np.inner(pca.components_[0, :], sbl_difference.values[i, :])

x = pd.Series(eigen)
conditions = {'KL≥3': cli['V00XRKL'] >= 3,
              'JSN Medial≥1': cli['V00XRJSM'] >= 1,
              'JSN Lateral≥1': cli['V00XRJSL'] >= 1,}

outcomes = {'Moderate Pain:': ((cli['V00WOMKP#'] >= 4) & (cli['V00WOMKP#'] < 8)),
            'Severe Pain:': (cli['V00WOMKP#'] >= 8),
            'Moderate Disability:': ((cli['V00WOMADL#'] >= 22) & (cli['V00WOMADL#'] < 45)),
            'Severe Disability:': (cli['V00WOMADL#'] >= 45),
            'Future TKR': (cli['V99E#KRPSN'] >= 1)}

for outcome in outcomes.keys():
    for condition in conditions.keys():
        print(outcome + ' | ' + condition)
        get_OR(x=x, outcome=outcomes[outcome],
               condition=conditions[condition])
    print('')


