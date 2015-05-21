import time

import os
import numpy
import theano.tensor as T

from NNBP import NNBP, randW, zerosW, zerosBias

from _data import mnist_load
from _results import pickleResultsGen, unpickleResultsGen, plotConfusionMatrix, testCost_validError_relationPlot, costPlot, mnist_visualise
from _utils import prettyPrintDictionary, prettyPrintDictionaryToString, getTimeID



def nnbp__learn_earlystopping(experiment_path, meta_params, dataset_info):
    N_epoch                 = meta_params["N_epoch"]
    batch_size              = meta_params["batch_size"]
    learning_rate           = meta_params["learning_rate"]
    valid_epoch_frequency   = meta_params["valid_epoch_frequency"]
    n_hl                    = meta_params["n_hl"]


    #######
    #Pre
    #######
    print '... loading data'
    dataset  = mnist_load()
    n_batches = {"train": (dataset["train"]["N"]) / batch_size, "valid": (dataset["valid"]["N"]) / batch_size, "test": (dataset["test"]["N"]) / batch_size}


    #######
    #Build
    #######
    print '... building the model'
    ibatch = T.lscalar()
    x = T.matrix('x')
    N_x = dataset_info["N_x"]
    y = T.ivector('y')
    N_y = dataset_info["N_y"]
    g = T.nnet.sigmoid
    o = T.nnet.softmax

    rng = numpy.random.RandomState(1234)
    initParamsFun = {"W_hl": randW, "b_hl": zerosBias, "W_ol": zerosW, "b_ol": zerosBias}
    nnbp_ml = NNBP(x, N_x, y, N_y, n_hl, g, o, rng, initParamsFun)

    fun_train, fun_valid, fun_test = nnbp_ml.compileFunctions(dataset, ibatch, batch_size, learning_rate)


    #######
    #Train
    #######
    print '... training'
    costs_test   = numpy.zeros((N_epoch), dtype=float)
    errors_valid = numpy.zeros((N_epoch/valid_epoch_frequency), dtype=int)

    start_time = time.clock()
    ivalid = 0
    iepoch = 0
    ifShouldEarlyStop = False
    while iepoch < N_epoch and ifShouldEarlyStop == False:
        for _ibatch in xrange(n_batches["train"]):
            costs_test[iepoch] += fun_train(_ibatch)

        if iepoch % valid_epoch_frequency == 0:
            for _ibatch in xrange(n_batches["valid"]):
                errors_valid[ivalid] += fun_valid(_ibatch)
            if iepoch > 5 and ivalid > 0 and errors_valid[ivalid] > errors_valid[ivalid - 1] and errors_valid[ivalid-1] > errors_valid[ivalid - 2]:
                ifShouldEarlyStop = True
            else:
                ivalid += 1

        print 'Epoch %d complete.' % (iepoch)
        iepoch += 1
    end_time = time.clock()

    learning_time = end_time - start_time
    print 'Optimization complete.' + 'Ran for %.2fm' % ((learning_time) / 60.)


    #######
    #Test on not-seen data
    #######
    print '... testing on not-seen data'
    error_test = 0
    y_pred = numpy.array([])
    for _ibatch in xrange(n_batches["test"]):
        errors_, y_pred_ = fun_test(_ibatch)
        error_test += errors_
        y_pred = numpy.concatenate((y_pred, y_pred_), axis=0)


    #######
    #Save results to files (pickling)
    #######
    print '... preparing results files :)'

    #Create info txt file
    infoFile = open(experiment_path + "/" + "info.txt", 'w')
    infoFile.write( "classification_method_name: neural network with back-propagation (early stopping)\n")
    infoFile.write( prettyPrintDictionaryToString("meta_params",  meta_params  ) )
    infoFile.write("\n")
    infoFile.write( prettyPrintDictionaryToString("dataset_info", dataset_info ) )
    infoFile.write("\n")
    infoFile.write( prettyPrintDictionaryToString("learning_info", {"error_test[%]": (float(error_test))/(float(dataset["test"]["N"]))*100, "learning_time [minutes]": learning_time/60., "N_epoch_actual": iepoch}) )
    infoFile.close()

    #pickle results
    pickleResultsGen(experiment_path + "/" + "infoPickled", [dataset_info, meta_params])
    nnmp_params_numpy = []
    for i in xrange(nnbp_ml.L+1):
        nnmp_params_numpy.append (nnbp_ml.params[i].get_value(borrow=True))
    results = {"costs_test": costs_test,
               "errors_valid": errors_valid,
               "error_test": error_test,
               "y_pred": y_pred,
               "learning_time": learning_time,
               "nnmp_params_numpy": nnmp_params_numpy,
               "N_epoch_actual": iepoch}
    pickleResultsGen(  experiment_path + "resultsPickled",  [dataset_info, meta_params, results] )

def nnbp__results(experiment_path):
    dataset  = mnist_load(ifTheanoTensorShared=False)
    dataset_info, meta_params, results = unpickleResultsGen(experiment_path + "/resultsPickled")

    #PrintDescription (name of dataset)(N_train, N_valid, N_test)(N_epoch, batch_size, learning_rate, valid_epoch_frequency)
    print 'Method name: back-propagation multilayer neural network (trained with early-stopping)'
    print 'Dataset: MNIST'
    print 'Meta params:'
    prettyPrintDictionary(meta_params)

    #Visualise data
    #mnist_visualise ( dataset["train"]["X"], (0, 500), "TRAIN" )

    #Learning plot: costs_test(iepoch), errors_valid(iepoch)
    costPlot(results["costs_test"],    experiment_path + "/test_costs.pdf")
    costPlot(results["errors_valid"],  experiment_path + "/valid_errors.pdf")
    #testCost_validError_relationPlot (costs_test, errors_valid)

    #Prediction error on test_set
    x = dataset["test"]["X"]
    N = len(x)
    y = dataset["test"]["Y"]
    y_pred = results["y_pred"]
    N_errors = sum(y_pred != y)
    print 'Errors rate on testing set: %f%% (%d of %d were wrongly classified)' % (float(N_errors)/float(N) * 100, N_errors, N)

    #Matrix of confusion on test set
    plotConfusionMatrix (x, N, y, y_pred, experiment_path + "/test_confusionMatrix.pdf")



def runExperiments():
    meta_params = {"N_epoch": 500, "batch_size": 1, "learning_rate": 0.01, "valid_epoch_frequency": 1, "n_hl": [500, 500, 500] }
    datasetInfo = {"name": "mnist", "load_fun": mnist_load, "N_x": 28*28, "N_y": 10}
    runExperiment(meta_params=meta_params, datasetInfo=datasetInfo)

def runExperiment(meta_params, datasetInfo):
    experimentID = getTimeID()

    #Create Folder
    pathToMainDirectory = "__results/nnbp"
    pathToExperimentDirectory = pathToMainDirectory +'/'+ experimentID
    os.makedirs(pathToExperimentDirectory)

    #######
    #Run experiment
    #######
    nnbp__learn_earlystopping(pathToExperimentDirectory+"/", meta_params, datasetInfo)



if __name__ == '__main__':
    #runExperiments()
    nnbp__results("__results/nnbp/"+"14-05-2015__08-35-14")