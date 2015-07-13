__author__ = 'Kadlubek47'

import time

import os
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from _data import mnist_load, prepareExamplesAsShared
from _results import \
    pickleResultsGen, unpickleResultsGen,\
    costPlot, mnist_visualise, weightsAsFilters
from _utils import \
    prettyPrintDictionary, prettyPrintDictionaryToString, \
    getTimeID, seperateLine, prepareSmallTrees

from SAE import SAE
from NNBP import randW, zerosBias, zerosW


def prettyPrintDictionaryToString(title, dict, tabsNum = 0):
    tabs = ''
    for i in xrange(tabsNum):
        tabs += '\t'
    #
    keys = dict.keys()
    values = dict.values()
    N = len(keys)
    sums = ""
    sums += tabs + title + ":\n"
    for i in xrange(N):
        if not (type(values[i]) == type({})):
            sums+= \
                tabs + "\t" + str(keys[i]) \
                + ": " + str(values[i]) + "\n"
        else:
            sums+= \
                prettyPrintDictionaryToString\
                (str(keys[i]), values[i], tabsNum+1)
    return sums


if __name__ == '__main__':
    ###########
    #Random number generator
    ###########
    rng = np.random.RandomState(15)
    T_rng = RandomStreams(rng.randint(2 ** 20))


    ###########
    #Prepare default meta_params
    ###########
    initP = \
        {'W_hl': randW,
        'b_hl' : zerosBias,
        'W_ol' : randW,
        'b_ol' : zerosBias}

    meta_params = \
        {'n_hl': [500, 500, 500],
        'pt':
            {'g': T.nnet.sigmoid,
            'o':  T.nnet.sigmoid,
            'corrupt': [0.3, 0.3, 0.3],
            ###
            'Ne': 15,
            'initP': initP,
            'B': 1,
            'K': 0.01},
        'ft':
            {'g': T.nnet.sigmoid,
            'o':  T.nnet.softmax,
            ###
            'Ne': 300,
            'initP': initP,
            'B': 1,
            'K': 0.01,
            'VF': 1},
        'rng': rng,
        'T_rng': T_rng}

    print prettyPrintDictionaryToString('meta_params', meta_params)

