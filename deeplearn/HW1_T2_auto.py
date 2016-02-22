##cell 0
#%pylab inline
#
##cell 1
#%load_ext autoreload
#%autoreload 2
#
#cell 2
import numpy
import pandas as pd

import theano
import theano.tensor as T

#cell 3
data = pd.read_pickle('/home/hudson/experiments/prostate.df')
#print data.head(2)
#print data[data.columns[-1]]

train_set_y = (data.values[:, -1] == 'tumor')
train_set_x = data.values[:, :-1]
n_genes = train_set_x.shape[1]
print n_genes
print train_set_x.shape
print train_set_y.shape
train_set_x = train_set_x.astype(float)
#train_set_y = train_set_y.astype(float)
print train_set_y

batch_size = 10
n_batches = len(train_set_y)/batch_size
print len(train_set_y), n_batches

#cell 4
# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(n_genes), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())


# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1

prediction = p_1 > 0.5                    # The prediction thresholded
#xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
#cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
cost = T.sum(-y * T.log(p_1) - (1-y) * T.log(1-p_1))
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)



index = T.lscalar()                
# Compile
learning_rate = 1e-2

train = theano.function(
          inputs=[x,y],
          outputs=cost,
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)



#cell 5
# Train
training_steps = 5
for i in range(training_steps):     
    cost = train_model(train_set_x, train_set_y)
    print i, T.cast(cost, 'int32').eval()
    
print("Final model:")
print(w.get_value(), w.get_value().size)
print(b.get_value(), b.get_value().size)
print("target values for Data:")
print(train_set_y)
print("prediction on Data:")
print(predict(train_set_x))

#cell 6

# Train

ep = range(10)
cs = []
for epoch in ep:
    c = 0
    for i in range(n_batches):
        c += train_model(i)
        pred, err = train_model(train_set_x, train_set_y)
    cs.append(c)

#cell 7



#cell 8


#cell 9


#cell 10


#cell 11


#cell 12





#cell 13


#cell 14


#cell 15


#cell 16


#cell 17


#cell 18



#cell 19
#class LogisticRegression(object):
#    
#    def __init__(self, input, n_in, n_out):
#        self.input = input
#        
#        W_values = 4*numpy.random.uniform(
#                low=-numpy.sqrt(6. / (n_in + n_out)),
#                high=numpy.sqrt(6. / (n_in + n_out)),
#                size=(n_in, n_out)
#        )
#        self.W = theano.shared(value=W_values, name='W', borrow=True)
#
#        b_values = numpy.zeros((n_out,))
#        self.b = theano.shared(value=b_values, name='b', borrow=True)
#           

#cell 20

        

#cell 21


#cell 22


#cell 23


#cell 24


