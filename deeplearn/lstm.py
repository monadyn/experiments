
import numpy
import theano
import theano.tensor as T

import gzip, os
import cPickle as pickle

def load_data(path, n_words, n_sample=5000, maxlen=1000):

    f = gzip.open(os.path.join(path, 'imdb.dict.pkl.gz'), 'rb')
    d = pickle.load(f)
    f.close()
    f = gzip.open(os.path.join(path, 'imdb.pkl.gz'), 'rb')
    train = pickle.load(f)
    test = pickle.load(f)
    f.close()

    dc = dict([(v, 0) for k, v in d.items()])
    for r in train[0]:
        for w in r:
            dc[w] += 1
    sdc = sorted(dc.items(), key=lambda x: x[1])[-1:0:-1]
    wds = [w[0] for w in sdc[:(n_words-1)]]
    wds = set(wds)
    tt = dict([(w, i+1) for i, w in enumerate(wds)])
    ds = [(k, tt[v]) for k, v in d.items() if v in wds]
    ds = dict(ds)

    def rmunk(data):
        return [[tt[w] if w in wds else 0 for w in sen] for sen in data]

    train = (rmunk(train[0]), train[1])
    test = (rmunk(test[0]), test[1])

    def subsample(x, y, n_samples, maxlen):
        lengths = numpy.array([len(s) for s in x])
        idx = numpy.nonzero(lengths <= maxlen)[0]
        numpy.random.shuffle(idx)
        idx = idx[:n_samples]
        x = [x[t] for t in idx]
        y = [y[t] for t in idx]
        return (x, y)

    train = subsample(train[0], train[1], n_sample, maxlen)
    test = subsample(test[0], test[1], n_sample, maxlen)

    return train, test, ds


def prepare_data(seqs, labels):
    n_samples = len(seqs)
    lengths = [len(s) for s in seqs]
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype(numpy.int64)
    x_mask = numpy.zeros((maxlen, n_samples)).astype(numpy.float64)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, numpy.array(labels)


def batches_idx(n, batch_size):

    idx_list = numpy.arange(n, dtype="int32")

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + batch_size])
        minibatch_start += batch_size
        
    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return minibatches


class lstm_layer:

    def __init__(self, n_hidden):
        
        def ortho_weight(ndim):
            W = numpy.random.randn(ndim, ndim)
            u, s, v = numpy.linalg.svd(W)
            return u
        
        self.n_hidden = n_hidden
        self.W = theano.shared(numpy.concatenate([ortho_weight(n_hidden),
                           ortho_weight(n_hidden),
                           ortho_weight(n_hidden),
                           ortho_weight(n_hidden)], axis=1))
        self.U = theano.shared(numpy.concatenate([ortho_weight(n_hidden),
                           ortho_weight(n_hidden),
                           ortho_weight(n_hidden),
                           ortho_weight(n_hidden)], axis=1))
        self.b = theano.shared(numpy.zeros((4 * n_hidden,)))    

        self.params = [self.W, self.U, self.b]
        
        
    def calc_lstm(self, input, mask):

        def _slice(_x, n, dim):
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = T.dot(h_, self.U)
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, self.n_hidden))
            f = T.nnet.sigmoid(_slice(preact, 1, self.n_hidden))
            o = T.nnet.sigmoid(_slice(preact, 2, self.n_hidden))
            c = T.tanh(_slice(preact, 3, self.n_hidden))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        n_samples = input.shape[1]

        wx = T.dot(input, self.W) + self.b
        rval, updates = theano.scan(_step,
                                sequences=[mask, wx],
                                outputs_info=[T.alloc(numpy.asarray(0., dtype=numpy.float64), 
                                                      n_samples, self.n_hidden),
                                              T.alloc(numpy.asarray(0., dtype=numpy.float64), 
                                                      n_samples, self.n_hidden)])

        return rval[0]


def rmsprop(lr, params, grads, x, mask, y, cost):

    zipped_grads = [theano.shared(p.get_value() * numpy.float64(0.)) for p in params]
    running_grads = [theano.shared(p.get_value() * numpy.float64(0.)) for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy.float64(0.)) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rgup + rg2up)

    updir = [theano.shared(p.get_value() * numpy.float64(0.)) for p in params]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(params, updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update




