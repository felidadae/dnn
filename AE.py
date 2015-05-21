
import theano
import theano.tensor as T


class AE(object):
    """
    Auto-Encoder class
    """

    def __init__(self, x, N_x, N_hl, g, o, rng, theano_rng, initParamsFun, W_hl=None, b_hl=None, b_ol=None):
        self.x = x
        if not W_hl: W_hl = initParamsFun["W_hl"](rng, N_x, N_hl, g)
        if not b_hl: b_hl = initParamsFun["b_hl"](N_hl)
        if not b_ol: b_ol = initParamsFun["b_ol"](N_x)
        self.W_hl = W_hl
        self.W_ol = W_hl.T
        self.b_hl = b_hl
        self.b_ol = b_ol
        self.g = g
        self.o = o
        self.rng = rng
        self.theano_rng = theano_rng

    def compileFunctions(self, x_image_global, dataset, ibatch, batch_size, learning_rate, corruption_level):
        if x_image_global == None:
            x_image_global = self.x

        train_set_x = dataset["train"]["X"]

        if corruption_level == 0.0:
            self.x_c = self.x
        else:
            self.x_c = self.theano_rng.binomial(size=self.x.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX) * self.x

        self.h = self.g(T.dot(self.x_c, self.W_hl) + self.b_hl)
        self.x_r = self.o(T.dot(self.h, self.W_ol) + self.b_ol)
        self.params = [self.W_hl, self.b_hl, self.b_ol]
        self.cost =   (- T.sum(self.x * T.log(self.x_r) + (1 - self.x) * T.log(1 - self.x_r), axis=(0,1)))

        gparams = T.grad(self.cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        fun_train = theano.function(
            inputs=[ibatch],
            outputs=(self.cost, self.x_r, self.x_c),
            updates=updates,
            givens={
                x_image_global: train_set_x[ibatch * batch_size: (ibatch + 1) * batch_size]
            }
        )

        return fun_train






