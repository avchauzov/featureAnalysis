from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, ShuffleSplit


def getFeaturesCorrelationRegression(_dataSet, _targetName):
	importanceList = []
	shuffleSplit = ShuffleSplit(n_splits = 10, test_size = 0.25)
	
	trainXData, testXData = pd.DataFrame(), pd.DataFrame()
	for trainIndices, testIndices in shuffleSplit.split(data):
		trainXData = pd.concat((trainXData, data.iloc[trainIndices]))
		testXData = pd.concat((testXData, data.iloc[testIndices]))
	
	modelsList = []
	for numberOfDistributions in range(1, 11):
		modelsList.append(GaussianMixture(numberOfDistributions).fit(trainXData['target'].values.reshape(-1, 1)))
	
	aicScores = [m.aic(trainXData['target'].values.reshape(-1, 1)) for m in modelsList]
	trainXData['target[AIC]'] = np.argmax(modelsList[aicScores.index(np.min(aicScores))].predict_proba(trainXData['target'].values.reshape(-1, 1)), axis = 1)
	testXData['target[AIC]'] = np.argmax(modelsList[aicScores.index(np.min(aicScores))].predict_proba(testXData['target'].values.reshape(-1, 1)), axis = 1)
	
	bicScores = [m.bic(trainXData['target'].values.reshape(-1, 1)) for m in modelsList]
	trainXData['target[BIC]'] = np.argmax(modelsList[bicScores.index(np.min(bicScores))].predict_proba(trainXData['target'].values.reshape(-1, 1)), axis = 1)
	testXData['target[BIC]'] = np.argmax(modelsList[bicScores.index(np.min(bicScores))].predict_proba(testXData['target'].values.reshape(-1, 1)), axis = 1)
	
	for feature in sorted([value for value in list(data) if value not in ['target']]):
		trainXDataAIC = deepcopy(trainXData[[feature, 'target[AIC]']])
		testXDataAIC = deepcopy(testXData[[feature, 'target[AIC]']])
		trainXDataAICGroup = trainXDataAIC.groupby('target[AIC]').agg(np.mean)
		testXDataAICGroup = testXDataAIC.groupby('target[AIC]').agg(np.mean)
		
		trainXDataBIC = deepcopy(trainXData[[feature, 'target[BIC]']])
		testXDataBIC = deepcopy(testXData[[feature, 'target[BIC]']])
		trainXDataBICGroup = trainXDataBIC.groupby('target[BIC]').agg(np.mean)
		testXDataBICGroup = testXDataBIC.groupby('target[BIC]').agg(np.mean)
		
		try:
			importanceList.append((feature,
			                       np.round(scipy.stats.pearsonr(trainXDataAICGroup[feature], testXDataAICGroup[feature])[0], 7),
			                       np.round(scipy.stats.pearsonr(trainXDataBICGroup[feature], testXDataBICGroup[feature])[0], 7)))
		
		except Exception as _:
			importanceList.append((feature, np.nan))
	
	return importanceList


def getFeaturesCorrelationClassification(_dataSet, _targetName):
	importanceList = []
	shuffleSplit = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10)
	
	trainXData, testXData = pd.DataFrame(), pd.DataFrame()
	for trainIndices, testIndices in shuffleSplit.split(data):
		trainXData = pd.concat((trainXData, data.iloc[trainIndices]))
		testXData = pd.concat((testXData, data.iloc[testIndices]))
	
	for feature in sorted([value for value in list(data) if value not in ['target']]):
		trainXData = deepcopy(trainXData[[feature, 'target']])
		testXData = deepcopy(testXData[[feature, 'target']])
		trainXDataGroup = trainXData.groupby('target').agg(np.mean)
		testXDataGroup = testXData.groupby('target').agg(np.mean)
		
		try:
			importanceList.append((feature,
			                       np.round(scipy.stats.pearsonr(trainXDataGroup[feature], testXDataGroup[feature])[0], 7),
			                       np.round(scipy.stats.pearsonr(trainXDataGroup[feature], testXDataGroup[feature])[0], 7)))
		
		except Exception as _:
			importanceList.append((feature, np.nan))
	
	return importanceList


NUMBEROFROWS = 1000
NUMBEROFCOLUMNS = 100

data = np.random.rand(NUMBEROFROWS, NUMBEROFCOLUMNS)
data = pd.DataFrame(data, columns = ['feature[' + str(index) + ']' for index in range(1, NUMBEROFCOLUMNS + 1)])

data['target'] = np.random.rand(NUMBEROFROWS)
# data['target'] = np.random.randint(0, 4)

featureImportances = getFeaturesCorrelationRegression(data, 'target')
# featureImportances = getFeaturesCorrelationClassification(data, 'target')

aicValues = sorted(list(set([value[1] for value in featureImportances])))
bicValues = sorted(list(set([value[2] for value in featureImportances])))

featureImportances = [(value[0], aicValues.index(value[1]) + bicValues.index(value[2])) for value in featureImportances]

featureImportancesDictionary = {}
for value in featureImportances:
	
	if value[1] in featureImportancesDictionary.keys():
		featureImportancesDictionary[value[1]].append(value[0])
	
	else:
		featureImportancesDictionary[value[1]] = [value[0]]

featureImportances = [(key, value) for key, value in featureImportancesDictionary.items()]
featureImportances.sort(key = lambda tup: tup[0], reverse = True)

scoresList = []
columnsToRemove = []

xData = data[[value for value in list(data) if value not in ['target'] + columnsToRemove]]
yData = data['target']

model = LinearRegression()
cvScores = cross_val_score(model, xData, yData, cv = ShuffleSplit(n_splits = 10, test_size = 0.25), scoring = 'neg_mean_squared_error')

# print(columnsToRemove, np.mean(cvScores), np.std(cvScores))

scoresList.append(np.mean(cvScores))

for _, column in featureImportances:
	columnsToRemove.extend(column)
	
	xData = data[[value for value in list(data) if value not in ['target'] + columnsToRemove]]
	yData = data['target']
	
	if len(list(xData)) == 0:
		continue
	
	model = LinearRegression()
	cvScores = cross_val_score(model, xData, yData, cv = ShuffleSplit(n_splits = 10, test_size = 0.25), scoring = 'neg_mean_squared_error')
	
	# print(columnsToRemove, np.mean(cvScores), np.std(cvScores))
	
	scoresList.append(np.mean(cvScores))

plt.plot(scoresList)
plt.show()
