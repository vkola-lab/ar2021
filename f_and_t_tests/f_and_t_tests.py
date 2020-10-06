import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

#grab knees w/ ONLY JSM. no JSL
JSM_ID = JSN.loc[JSN.M_grade > 0.0]
JSM_ID = JSM_ID.loc[JSM_ID.L_grade == 0.0]

#grab knees w/ ONLY no JSN.
noJSN_ID = JSN.loc[JSN.M_grade == 0.0]
noJSN_ID = noJSN_ID.loc[noJSN_ID.L_grade == 0.0]



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

	for (JSNName, JSNData) in dataN.iteritems():
		t_yes = ''
		f_yes = ''
		noJSNData = dataNo.loc[:, JSNName]
		t, p_t = stats.ttest_ind(JSNData, noJSNData)
		f, p_f = f_test(JSNData, noJSNData)

		if p_t < 0.05:
			t_yes = 'significant'
		if p_f < 0.05:
			f_yes = 'significant'

		new_row = {'t':t, 
			'p_t': p_t, 
			'p_t_sig': t_yes, 
			'f':f, 
			'p_f':p_f,
			'p_f_sig': f_yes}

		result = result.append(new_row, ignore_index=True)
	result.to_csv(fileName)

result_tibia_lateral = runTandFTest(tibia_JSL, tibia_noJSN, 'tibia_lateral.csv')
result_femur_lateral = runTandFTest(femur_JSL, femur_noJSN, 'femur_lateral.csv')
result_femur_lateral = runTandFTest(femur_JSM, femur_noJSN, 'femur_medial.csv')
result_tibia_medial = runTandFTest(tibia_JSL, tibia_noJSN, 'tibia_medial.csv')


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