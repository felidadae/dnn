"""
For classes HiddenLayer, OutputLayer, NNBP_ml:
variables with names: h_p, h, W, b, g, o, x, y are symbolic (Theano.Tensor)
"""
import numpy
import theano
import theano.tensor as T



def randW(rng, N_h_p, N_h, g):
    W_values = numpy.asarray(
        rng.uniform(
            low=-numpy.sqrt(6. / (N_h_p + N_h)),
            high=numpy.sqrt(6. / (N_h_p + N_h)),
            size=(N_h_p, N_h)
        ),
        dtype=theano.config.floatX
    )
    if g == theano.tensor.nnet.sigmoid:
        W_values *= 4
    return theano.shared(value=W_values, name='W', borrow=True)

def zerosW(N_h_p, N_h):
    return theano.shared(
        value=numpy.zeros(
            (N_h_p, N_h),
            dtype=theano.config.floatX
        ),
        name='W',
        borrow=True
    )

def zerosBias(N_h):
    b_values = numpy.zeros((N_h,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    return b



class HiddenLayer(object):
    def __init__(self, h_p, N_h_p, N_h, W, b, g):
        self.h_p = h_p
        self.N_h_p = N_h_p
        self.N_h = N_h
        self.g = g
        self.W = W
        self.b = b
        self.h = g(T.dot(h_p, W) + b)

class OutputLayer(object):
    softmax = T.nnet.softmax
    def __init__(self, h_p, N_h_p, N_y, W, b, o):
        self.h_p = h_p
        self.N_h_p = N_h_p
        self.N_y = N_y
        self.W = W
        self.b = b
        self.o = o

        self.f = o(T.dot(h_p,W) + b)

class NNBP(object):
    def __init__(self, x, N_x, y, N_y, n_hl, g, o, rng, initP):
        self.x = x
        self.y = y

        #Hidden layers
        self.hiddenLayers = []
        self.params = []
        self.L = len(n_hl)
        for ih in xrange(self.L):
            if ih==0:
                N_h_p = N_x
                N_h = n_hl[ih]
                h_p = x
            else:
                N_h_p = n_hl[ih-1]
                N_h  = n_hl[ih]
                h_p = self.hiddenLayers[ih-1].h
            W = initP["W_hl"](rng, N_h_p, N_h, g)
            b = initP["b_hl"](N_h)
            self.params.append(W)
            self.params.append(b)
            layer = \
                HiddenLayer(h_p, N_h_p, N_h, W, b, T.nnet.sigmoid)
            self.hiddenLayers.append(layer)

        #Output layer
        h_p  = self.hiddenLayers[self.L-1].h
        N_h_p = n_hl[self.L-1]
        W = initP["W_ol"](rng, N_h_p, N_y, o)
        b = initP["b_ol"](N_y)
        self.params.append(W)
        self.params.append(b)
        self.outputLayer = \
            OutputLayer(h_p, N_h_p, N_y, W, b, o)
        self.f = self.outputLayer.f

        #y_pred, cost, errors, params
        self.y_pred =  T.argmax(self.f, axis=1)
        self.cost   = -T.mean(T.log(self.f)[T.arange(y.shape[0]), y])
        #self.cost = T.sum( T.pow(self.f - y, 2))
        self.errors =  T.sum(T.neq(self.y_pred, y))

    def compileFunctions (self, dataset, ib, B, K):
        train_set_x = dataset["train"]["X"]
        train_set_y = dataset["train"]["Y"]
        valid_set_x = dataset["valid"]["X"]
        valid_set_y = dataset["valid"]["Y"]
        test_set_x  = dataset["test" ]["X"]
        test_set_y  = dataset["test" ]["Y"]

        gparams = [T.grad(self.cost, param) for param in self.params]
        updates = [
            (param, param - K * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        fun_train = theano.function(
            inputs=[ib],
            outputs=self.cost,
            updates=updates,
            givens={
                self.x: train_set_x[ib * B: (ib + 1) * B],
                self.y: train_set_y[ib * B: (ib + 1) * B]
            }
        )

        fun_validate = theano.function(
            inputs=[ib],
            outputs=self.errors,
            givens={
                self.x: valid_set_x[ib * B:(ib + 1) * B],
                self.y: valid_set_y[ib * B:(ib + 1) * B]
            }
        )

        fun_test = theano.function(
            inputs=[ib],
            outputs=(self.errors, self.y_pred),
            givens={
                self.x: test_set_x[ib * B:(ib + 1) * B],
                self.y: test_set_y[ib * B:(ib + 1) * B]
            }
        )

        return fun_train, fun_validate, fun_test