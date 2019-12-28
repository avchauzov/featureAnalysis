import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, ShuffleSplit


def getFeaturesCorrelationRegression(_dataSet, _targetName):
	importanceList = []
	shuffleSplit = ShuffleSplit(n_splits = 10, test_size = 0.25)
	
	trainXData, testXData = pd.DataFrame(), pd.DataFrame()
	for trainIndices, testIndices in shuffleSplit.split(data):
		trainXData = pd.concat((trainXData, data.iloc[trainIndices]))
		testXData = pd.concat((testXData, data.iloc[testIndices]))
	
	for feature in sorted([value for value in list(data) if value != 'target']):
		modelsList = []
		
		for numberOfDistributions in range(1, 4):
			modelsList.append(GaussianMixture(numberOfDistributions).fit(trainXData[feature].values.reshape(-1, 1)))
		
		aicScores = [m.aic(trainXData[feature].values.reshape(-1, 1)) for m in modelsList]
		
		trainXDataAIC = deepcopy(trainXData[[feature, 'target']])
		testXDataAIC = deepcopy(trainXData[[feature, 'target']])
		
		trainXDataAIC[feature] = np.argmax(modelsList[aicScores.index(np.min(aicScores))].predict_proba(trainXDataAIC[feature].values.reshape(-1, 1)), axis = 1)
		testXDataAIC[feature] = np.argmax(modelsList[aicScores.index(np.min(aicScores))].predict_proba(testXDataAIC[feature].values.reshape(-1, 1)), axis = 1)
		
		trainXDataAICGroup = trainXDataAIC.groupby(feature).agg(np.mean)
		testXDataAICGroup = testXDataAIC.groupby(feature).agg(np.mean)
		
		bicScores = [m.bic(trainXData[feature].values.reshape(-1, 1)) for m in modelsList]
		
		trainXDataBIC = deepcopy(trainXData[[feature, 'target']])
		testXDataBIC = deepcopy(trainXData[[feature, 'target']])
		
		trainXDataBIC[feature] = np.argmax(modelsList[bicScores.index(np.min(bicScores))].predict_proba(trainXDataBIC[feature].values.reshape(-1, 1)), axis = 1)
		testXDataBIC[feature] = np.argmax(modelsList[bicScores.index(np.min(bicScores))].predict_proba(testXDataBIC[feature].values.reshape(-1, 1)), axis = 1)
		
		trainXDataBICGroup = trainXDataBIC.groupby(feature).agg(np.mean)
		testXDataBICGroup = testXDataBIC.groupby(feature).agg(np.mean)
		
		try:
			importanceList.append((feature,
			                       np.round(scipy.stats.pearsonr(trainXDataAICGroup['target'], testXDataAICGroup['target'])[0] + np.random.uniform(), 3),
			                       np.round(scipy.stats.pearsonr(trainXDataBICGroup['target'], testXDataBICGroup['target'])[0] + np.random.uniform(), 3)))
		
		except Exception as _:
			importanceList.append((feature, np.nan))
	
	return importanceList


NUMBEROFROWS = 1000
NUMBEROFCOLUMNS = 100

data = np.random.rand(NUMBEROFROWS, NUMBEROFCOLUMNS)
data = pd.DataFrame(data, columns = ['feature[' + str(index) + ']' for index in range(1, NUMBEROFCOLUMNS + 1)])
data['target'] = np.random.rand(NUMBEROFROWS)

featureImportances = getFeaturesCorrelationRegression(data, 'target')

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
