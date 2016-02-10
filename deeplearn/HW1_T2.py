import pandas as pd
from collections import Counter
import numpy as np

data = pd.read_pickle('prostate.df')
#print data.head(10)

y = data.values[:, -1]
print 'targets: ',y.shape, Counter(y.tolist())
x = data.values[:, :-1]
print 'features: ', x.shape


#spliit data
n_folds = 10
from sklearn.cross_validation import KFold
#10 fold cross-validation
kf = KFold(len(y), n_folds=n_folds)

import numpy 
import theano 
import theano.tensor as T 
rng = numpy.random
N = 10      # number of samples 
feats = 8  # dimensionality of features 
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))   
training_steps = 10 # x axis
#D = (x[0:10], y[0:10])
print (D)

#exit(1)
# declare Theano symbolic variables
x = T.matrix("x") 
y = T.vector("y") 
w = theano.shared(rng.randn(feats), name="w") 
b = theano.shared(0., name="b")  
print "Initial model:" 
print w.get_value(), b.get_value()  

# Construct Theano expression graph 
p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b)) # probability that target = 1 
prediction = p_1 > 0.5                # the prediction threshold

#neg. log likelihood, y axis
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1) # cross-entropy loss func 
cost = xent.mean() + 0.01 * (w**2).sum()  # the cost to minimize 
gw, gb = T.grad(cost, [w, b]) 

# Compile
learnRate = 0.1 
train = theano.function(inputs = [x, y], \
		outputs = [prediction, xent, cost], \
		updates = {w : w-learnRate*gw, b : b-learnRate*gb}) 
predict = theano.function(inputs = [x], outputs = prediction) 


# Train 
for i in range(training_steps):  
	pred, err, cost_val = train(D[0], D[1]) 
	print i, '-->', pred, err, cost_val 
#print "Final model:" 
#print w.get_value(), b.get_value() 
#print "target values for D: ", D[1] 
print "predictions on D: ", predict(D[0]) 


#from sklearn import datasets
#X, y = datasets.make_regression(int(1e6))
#X, y = datasets.make_regression(n_samples=1000, n_features=4)
#print X.shape

#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression()

for train_index, test_index in kf:
	X_train, X_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	#print X_train.shape, X_test.shape, y_train.shape, y_test.shape
	#train_errors[index],test_errors[index] = learn_and_measure(X_train, y_train, X_test, y_test, depth=d)
	
	#lr.fit(X_train, y_train)
	#y_train_pred = lr.predict(X_train)
	#y_test_pred = lr.predict(X_test)
	#print y_train_pred
	#print y_test_pred
		

exit(0)

##############################################################

import numpy 
import theano 
import theano.tensor as T 
rng = numpy.random
N = 400      # number of samples 
feats = 784  # dimensionality of features 
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))   
training_steps = 10

# declare Theano symbolic variables
x = T.matrix("x") 
y = T.vector("y") 
w = theano.shared(rng.randn(784), name="w") 
b = theano.shared(0., name="b")  
print "Initial model:" 
print w.get_value(), b.get_value()  

# Construct Theano expression graph 
p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b)) # probability that target = 1 
prediction = p_1 > 0.5                # the prediction threshold 
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1) # cross-entropy loss func 
cost = xent.mean() + 0.01 * (w**2).sum()  # the cost to minimize 
gw, gb = T.grad(cost, [w, b]) 

# Compile 
train = theano.function(inputs = [x, y],   outputs = [prediction, xent],   updates = {w : w-0.1*gw, b : b-0.1*gb}) 
predict = theano.function(inputs = [x], outputs = prediction) 


# Train 
for i in range(training_steps):  
	pred, err = train(D[0], D[1])  
print "Final model:" 
print w.get_value(), b.get_value() 
print "target values for D: ", D[1] 
print "predictions on D: ", predict(D[0]) 



exit(0)

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



