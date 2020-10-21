import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import scipy

plt.rcParams.update({'font.size': 14})

#slice_normalized = pd.Series(list(range(100)), index=index)
#Left knees first then right.

def dFFromFile(fileName):
	df_array = np.load(fileName)
	df = pd.DataFrame(df_array)
	return df

#grab data from files and create dataframes
LT = dFFromFile('LT_all.npy')
RT = dFFromFile('RT_all.npy')
LF = dFFromFile('LF_all.npy')
RF = dFFromFile('RF_all.npy')

JSL_left = dFFromFile('JSL_left.npy')
JSL_left = JSL_left.T
JSL_right = dFFromFile('JSL_right.npy')
JSL_right = JSL_right.T

JSM_left = dFFromFile('JSM_left.npy')
JSM_left = JSM_left.T

JSM_right = dFFromFile('JSM_right.npy')
JSM_right = JSM_right.T



#combine left and right knees into one dataframe. left top rows, right bottom rows.
tibia = LT.append(RT, ignore_index=True)
femur = LF.append(RF, ignore_index=True)

JSL = JSL_left.append(JSL_right, ignore_index=True)
JSL.columns = ['L_grade']
JSM = JSM_left.append(JSM_right, ignore_index=True)
JSM.columns = ['M_grade']

#combine JSL and JSM into one dataframe. lateral 1st col. medial right col.
JSN = pd.concat([JSL, JSM], axis=1)



#grab knees w/ ONLY JSL. no JSM
JSL_ID = JSN.loc[JSN.L_grade > 0.0]
JSL_ID = JSL_ID.loc[JSL_ID.M_grade == 0.0]
print ("Number of subjects with only lateral JSN = " + str(len(JSL_ID)))
print JSL_ID

#grab knees w/ ONLY JSM. no JSL
JSM_ID = JSN.loc[JSN.M_grade > 0.0]
JSM_ID = JSM_ID.loc[JSM_ID.L_grade == 0.0]
print ("Number of subjects with only medial JSN = " + str(len(JSM_ID)))
print JSM_ID

#grab knees w/ ONLY no JSN.
noJSN_ID = JSN.loc[JSN.M_grade == 0.0]
noJSN_ID = noJSN_ID.loc[noJSN_ID.L_grade == 0.0]
print ("Number of subjects with no JSN = " + str(len(noJSN_ID)))
print noJSN_ID

#grab subchondral lengths
#JSL only
tibia_JSL = tibia.loc[JSL_ID.index.values]
femur_JSL = femur.loc[JSL_ID.index.values]
#JSM only
tibia_JSM = tibia.loc[JSM_ID.index.values]
femur_JSM = femur.loc[JSM_ID.index.values]
#noJSN only
tibia_noJSN = tibia.loc[noJSN_ID.index.values]
femur_noJSN = femur.loc[noJSN_ID.index.values]


#define F-test function
def f_test(x, y):
	x = x.values.tolist()
	y = y.values.tolist()
	x = np.array(x)
	y = np.array(y)
	f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
	dfn = x.size-1 #define degrees of freedom numerator 
	dfd = y.size-1 #define degrees of freedom denominator 
	p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
	return f, p

#function that runs t and f test between dataset w/ JSN and dataset w/out JSN.
def runTandFTest(dataN, dataNo, fileName):
	results = {'t': [],
	'p_t': [],
	'p_t_sig':[],
	'f': [],
	'p_f': [],
	'p_f_sig': []}
	result = pd.DataFrame(results)
	slice = 0.0
	slice_list = []
	t_list = []
	p_t_list = []

	for (JSNName, JSNData) in dataN.iteritems():
		
		t_yes = ''
		f_yes = ''
		noJSNData = dataNo.loc[:, JSNName]
		t, p_t = stats.ttest_ind(JSNData, noJSNData, equal_var = False)
		f, p_f = f_test(JSNData, noJSNData)
		t = round(t,2)

		t_list.append(t)
		p_t_list.append(p_t)
		slice_list.append(slice/200)
		slice = slice + 1

		if p_t < 0.05:
			t_yes = 'significant'
		if p_f < 0.05:
			f_yes = 'significant'

		if p_t < 0.0001:
			p_t = 'p<0.0001'
		elif p_t < 0.001:
			p_t = 'p<0.001'
		elif p_t < 0.01:
			p_t = round(p_t,3)
		else:
			p_t = round(p_t,2)

		new_row = {'t':t, 
			'p_t': p_t, 
			'p_t_sig': t_yes, 
			'f':f, 
			'p_f':p_f,
			'p_f_sig': f_yes}

		result = result.append(new_row, ignore_index=True)

	result.to_csv(fileName)
	return (t_list, p_t_list, slice_list)

#function that plots T test results
def plotStatResults(t_list, p_t_list, slice_list, title):
	colors = []
	for p_t in p_t_list:
		if p_t < 0.0001:
			colors.append('darkred')
		elif p_t < 0.001:
			colors.append('red')
		elif p_t < 0.01:
			colors.append('orange')
		elif p_t <0.05:
			colors.append('yellow')
		else:
			colors.append('green')

	fig, ax = plt.subplots(figsize = (10,6))
	fig.figsize = (10,6)
	scatter = ax.scatter(slice_list, t_list, s=15, c=colors, alpha =0.5)
	color_legend = ['darkred', 'red', 'orange', 'yellow', 'green']
	lines = [Line2D([], [], color= c, markersize=3, linewidth=0, marker='o') for c in color_legend]
	labels = ['p<0.0001', 'p<0.001', 'p<0.01', 'p<0.05', 'p>=0.05']

	ax.legend(lines, labels, prop={'size':10})
	plt.title(title)
	plt.show()

#function that calculates difference between means (for each slice) of dataset with JSN vs dataset without JSN
def diffMeans(dataN, dataNo):
	diff_mean = []

	for (JSNName, JSNData) in dataN.iteritems():
		noJSNData = dataNo.loc[:, JSNName]

		noJSNData = noJSNData.values.tolist()
		JSNData = JSNData.values.tolist()
		noJSNData = np.array(noJSNData)
		JSNData = np.array(JSNData)

		mean_noJSN = np.mean(noJSNData)
		mean_JSN = np.mean(JSNData)

		diff = mean_JSN - mean_noJSN
		diff_mean.append(diff)

	return diff_mean

print ("running T and F test tibia JSL vs tibia no JSN")
t, p_t, slice = runTandFTest(tibia_JSL, tibia_noJSN, 'tibia_lateral.csv')
diff_mean = diffMeans(tibia_JSL, tibia_noJSN)

title = 'T test: Tibia with JSN Lateral vs. Tibia with no JSN'
plotStatResults(t, p_t, slice, title)

title = 'Difference Between Means: Tibia with JSN Lateral vs. Tibia with no JSN'
plotStatResults(diff_mean, p_t, slice, title)



print ("running T and F test femur JSL vs femur no JSN")
t, p_t, slice = runTandFTest(femur_JSL, femur_noJSN, 'femur_lateral.csv')
diff_mean = diffMeans(femur_JSL, femur_noJSN)

title = 'T test: Femur with JSN Lateral vs. Femur with no JSN'
plotStatResults(t, p_t, slice, title)

title = 'Difference Between Means: Femur with JSN Lateral vs. Femur with no JSN'
plotStatResults(diff_mean, p_t, slice, title)



print ("running T and F test femur JSM vs femur no JSN")
t, p_t, slice = runTandFTest(femur_JSM, femur_noJSN, 'femur_medial.csv')
diff_mean = diffMeans(femur_JSM, femur_noJSN)

title = 'T test: Femur with JSN Medial vs. Femur with no JSN'
plotStatResults(t, p_t, slice, title)

title = 'Difference Between Means: Femur with JSN Medial vs. Femur with no JSN'
plotStatResults(diff_mean, p_t, slice, title)



print ("running T and F test tibia JSM vs tibia no JSN")
t, p_t, slice = runTandFTest(tibia_JSM, tibia_noJSN, 'tibia_medial.csv')
diff_mean = diffMeans(tibia_JSM, tibia_noJSN)

title = 'T test: Tibia with JSN Medial vs. Tibia with no JSN'
plotStatResults(t, p_t, slice, title)

title = 'Difference Between Means: Tibia with JSN Medial vs. Tibia with no JSN'
plotStatResults(diff_mean, p_t, slice, title)

"""
plt.figure(); tibia_noJSN.boxplot()
top = plt.ylim()
plt.ylim(top=200)
plt.show()

plt.figure(); tibia_JSL.boxplot()
top = plt.ylim()
plt.ylim(top=200)
plt.show()

plt.figure(); tibia_JSM.boxplot()
top = plt.ylim()
plt.ylim(top=200)
plt.show()
"""
