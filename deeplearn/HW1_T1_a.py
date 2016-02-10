import pandas as pd
from collections import Counter
import numpy as np

data = pd.read_pickle('prostate.df')
#print data.head(10)

y = data.values[:, -1]
print 'targets: ',y.shape, Counter(y.tolist())
x = data.values[:, :-1]
print 'features: ', x.shape



from sklearn import tree
from sklearn import metrics
def learn_and_measure(X_train, y_train, X_test, y_test, depth=False, min_samples_leaf=False):
	if depth:
		clf=tree.DecisionTreeClassifier(max_depth=depth)
	elif min_samples_leaf:	
		clf=tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
	clf.fit(X_train,y_train)
	train_err = metrics.accuracy_score(y_train,clf.predict(X_train))
	test_err = metrics.accuracy_score(y_test,clf.predict(X_test))
	return train_err, test_err

	#clf=tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)


#spliit data
n_folds = 10
from sklearn.cross_validation import KFold
#10 fold cross-validation
kf = KFold(len(y), n_folds=n_folds)

#######################################################################################################################
# vary the depth of decision trees
max_depth = np.arange(1, 20)
avg_train_err = np.zeros(len(max_depth))
avg_test_err = np.zeros(len(max_depth))
for i, d in enumerate(max_depth):
	#print '\n',i,d
	train_errors = np.zeros(n_folds)
	test_errors = np.zeros(n_folds)
	index = 0
	for train_index, test_index in kf:
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		#print X_train.shape, X_test.shape, y_train.shape, y_test.shape
		train_errors[index],test_errors[index] = learn_and_measure(X_train, y_train, X_test, y_test, depth=d)
		index += 1
	avg_train_err[i] = np.mean(train_errors) 
	avg_test_err[i] = np.mean(test_errors)


print max_depth
print avg_train_err
print avg_test_err

import matplotlib as mpl 
mpl.use('Agg') 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import * 
import pylab as pl

# Start a new figure 
pl.figure()
pl.title('Decision Trees: training error and testing error v.s. tree depth')
pl.plot(max_depth, avg_test_err, lw=2, label = 'test error')
pl.plot(max_depth, avg_train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('tree depth')
pl.ylabel('error')
#pl.show()
pl.savefig('HW1_TASK1_a.png')



