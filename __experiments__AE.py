
import time

import os
import numpy
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from NNBP import randW, zerosBias
from AE import AE

from _results import pickleResultsGen, unpickleResultsGen, costPlot, mnist_visualise, weightsAsFilters
from _data import mnist_load
from _utils import seperateLine, prettyPrintDictionary, prettyPrintDictionaryToString, getTimeID



def ae__learn(experiment_path, meta_params, dataset_info):
    N_epoch                 = meta_params["N_epoch"]
    batch_size              = meta_params["batch_size"]
    learning_rate           = meta_params["learning_rate"]
    N_hl                    = meta_params["N_hl"]
    corruption_level        = meta_params["corruption_level"]


    #######
    #Pre
    #######
    print '... loading data'
    dataset  = dataset_info["load_fun"]()
    n_batches = {"train": (dataset["train"]["N"]) / batch_size, "valid": (dataset["valid"]["N"]) / batch_size, "test": (dataset["test"]["N"]) / batch_size}


    #######
    #Build
    #######
    print '... building the model'
    ibatch = T.lscalar()
    x = T.matrix('x')
    N_x = dataset_info["N_x"]
    y = T.ivector('y')
    g = T.nnet.sigmoid
    o = T.nnet.sigmoid

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    initParamsFun = {"W_hl": randW, "b_hl": zerosBias, "b_ol": zerosBias}
    ae = AE(x, N_x, N_hl, g, o, rng, theano_rng, initParamsFun)
    fun_train = ae.compileFunctions(x, dataset, ibatch, batch_size, learning_rate, corruption_level)


    #######
    #Train
    #######
    start_time = time.clock()
    cost   = numpy.zeros(N_epoch, dtype=float)
    cost_final = numpy.zeros((dataset["train"]["N"], N_x), dtype=dataset["train"]["X"].get_value(borrow=True).dtype)
    x_r_final  = numpy.zeros((dataset["train"]["N"], N_x), dtype=dataset["train"]["X"].get_value(borrow=True).dtype)
    x_c_final  = numpy.zeros((dataset["train"]["N"], N_x), dtype=dataset["train"]["X"].get_value(borrow=True).dtype)
    for iepoch in xrange(N_epoch):
        if iepoch < N_epoch-1:
            for _ibatch in xrange(n_batches["train"]):
                cost, _, _ = fun_train(_ibatch)
                cost[iepoch] += cost
        else:
            for _ibatch in xrange(n_batches["train"]):
                cost, x_r_part, x_c_part = fun_train(_ibatch)
                x_r_final[_ibatch*batch_size:(_ibatch+1)*batch_size] = x_r_part
                x_c_final[_ibatch*batch_size:(_ibatch+1)*batch_size] = x_c_part
                cost_final[_ibatch] = cost
                cost[iepoch] += cost
        print 'Training epoch %d, cost ' % iepoch, cost[iepoch]
    end_time = time.clock()
    learning_time = end_time - start_time
    print '\nOptimization complete.' + 'Ran for %.2fm' % (learning_time / 60.)


    #######
    #Save results to files (pickling)
    #######
    print '... preparing results files :)'

    #Create info txt file
    infoFile = open(experiment_path + "/" + "info.txt", 'w')
    infoFile.write( "method_name: autoencoder\n")
    infoFile.write( prettyPrintDictionaryToString("meta_params",  meta_params  ) )
    infoFile.write("\n")
    infoFile.write( prettyPrintDictionaryToString("dataset_info", dataset_info ) )
    infoFile.write("\n")
    infoFile.write( prettyPrintDictionaryToString("learning_info", {"cost_final": cost[N_epoch-1], "learning_time [minutes]": learning_time/60.}) )
    infoFile.close()
    
    #pickle results
    ae_params_numpy = [ae.W_hl.get_value(borrow=True), ae.b_hl.get_value(borrow=True)]
    results = {"ae_params_numpy": ae_params_numpy, "costs": cost, "x_r_final": x_r_final, "x_c_final": x_c_final}
    pickleResultsGen( experiment_path + "resultsPickled", [meta_params, results] )

def ae__results(experiment_path):
    dataset  = mnist_load(ifTheanoTensorShared=False)
    meta_params, results  = unpickleResultsGen(experiment_path + "/resultsPickled")
    cost           = results["costs"]
    x_r_final       = results["x_r_final"]
    ae_params_numpy = results["ae_params_numpy"]
    W_hl,b_hl = ae_params_numpy

    #PrintDescription (name of dataset)(N_train, N_valid, N_test)(N_epoch, batch_size, learning_rate, valid_epoch_frequency)
    seperateLine(before=True)
    print 'Method name: auto-encoder'
    print 'Dataset: MNIST'
    print 'Meta params:'
    prettyPrintDictionary(meta_params)
    seperateLine(after=True)

    #Learning plot: cost(ibatch)
    costPlot(cost, experiment_path + "/costplot.pdf")

    #Show predicted images
    mnist_visualise ( dataset["train"]["X"],    (0, 500), "ORIGINAL",       experiment_path + "/originalSet.png"        )
    mnist_visualise ( results["x_c_final" ],    (0, 500), "CORRUPTED",      experiment_path + "/corruptedSet.png"       )
    mnist_visualise ( x_r_final,                (0, 500), "RECONSTRUCTED",  experiment_path + "/reconstructedSet.png"   )

    #Filters from weights
    mnist_visualise(  W_hl.T, (0,meta_params["N_hl"]), "Weights as filters (W_hl.T)", experiment_path + "/weightsAsFilters.png")
    weightsAsFilters(W_hl.T, experiment_path + "/weightsAsFilters2.png")



def runExperiments():
    meta_params = {"N_epoch": 15, "batch_size": 1, "learning_rate": 0.01, "valid_epoch_frequency": 1, "N_hl": 60, "corruption_level": 0.5 }
    datasetInfo = {"name": "mnist", "load_fun": mnist_load, "N_x": 28*28, "N_y": 10}

    #4
    meta_params["N_hl"] = 1100
    meta_params["corruption_level"] = 0.5
    runExperiment(meta_params=meta_params, datasetInfo=datasetInfo)


def runExperiment (meta_params, datasetInfo):
    experimentID = getTimeID()

    #Create experiment folder
    pathToMainDirectory = "__results/ae"
    pathToExperimentDirectory = pathToMainDirectory +'/'+ experimentID
    os.makedirs(pathToExperimentDirectory)

    #######
    #Run experiment
    #######
    ae__learn(pathToExperimentDirectory+"/", meta_params, datasetInfo)



if __name__ == '__main__':
    #runExperiments()
    #ae__results  ("__results/ae/"+"16-05-2015__15-28-21")
    #ae__results  ("__results/ae/"+"16-05-2015__15-32-26")
    #ae__results  ("__results/ae/"+"16-05-2015__15-36-33")
    #ae__results  ("__results/ae/"+"16-05-2015__15-45-39")
    #ae__results  ("__results/ae/"+"16-05-2015__15-54-59")
    ae__results  ("__results/ae/"+"16-05-2015__19-03-00")

