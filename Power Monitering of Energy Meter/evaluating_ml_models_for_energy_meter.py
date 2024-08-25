# -*- coding: utf-8 -*-
"""Evaluating_ML_models_for_Energy_meter.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KxMbvJnvt4A0XF3gNu-rWXM3F8_4DF0W

# Energy meter using ML

## Loading Dataset
"""

from pandas import read_csv

filename = 'Energy Meter.csv'
names = ['Voltage','Current','Power','class']
data = read_csv(filename, names = names)

"""## Summarize Dataset"""

print(data.shape)
print(data.head(20))
print(data.describe())
print(data.groupby('class').size())

"""## Visualize Data"""

from pandas.plotting import scatter_matrix
from matplotlib import pyplot

data.plot(kind = 'box', subplots = True, layout = (2,2))
pyplot.title('BAR PLOT')
pyplot.show()

data.hist()
pyplot.title('HISTOGRAM')
pyplot.show()

scatter_matrix(data)
pyplot.title('SCATTER MATRIX')
pyplot.show()

"""##Evaluating various ML algorithm"""

# 6 ML Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

arrey = data.values
X = arrey[:,0:3]
y = arrey[:,3]
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.20, random_state = 1)

models = []
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma = 'auto')))

results = []
names = []
res = []
for name, model in models:
  kfold = StratifiedKFold(n_splits = 10, random_state = None)
  cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
  results.append(cv_results)
  names.append(name)
  res.append(cv_results.mean())
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.ylim(.990,.999)
pyplot.bar(names, res, color = ['black', 'red', 'green', 'blue', 'cyan', 'yellow'], width = 0.6)
pyplot.title('Algorithm Comparison')
pyplot.show()