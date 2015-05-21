import time

import os
import numpy
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from NNBP import randW, zerosW, zerosBias
from SAE import SAE

from _data import mnist_load
from _results import pickleResultsGen, unpickleResultsGen, plotConfusionMatrix, testCost_validError_relationPlot, costPlot, mnist_visualise
from _utils import prettyPrintDictionary, prettyPrintDictionaryToString, getTimeID, seperateLine



def sae__learn_earlystopping(experiment_path, meta_params, dataset_info):
    N_epoch                 = meta_params["N_epoch"]
    N_epoch_pretrain        = meta_params["N_epoch_pretrain"]
    batch_size              = meta_params["batch_size"]
    learning_rate           = meta_params["learning_rate"]
    pretrain_rate           = meta_params["pretrain_rate"]
    valid_epoch_frequency   = meta_params["valid_epoch_frequency"]
    n_hl                    = meta_params["n_hl"]
    corruption_levels       = meta_params["corruption_levels"]


    #######
    #Load data
    #######
    print '... loading data'
    dataset  = mnist_load()
    n_batches = {"train": (dataset["train"]["N"]) / batch_size, "valid": (dataset["valid"]["N"]) / batch_size, "test": (dataset["test"]["N"]) / batch_size}


    #######
    #Build the model
    #######
    print '... building the model'
    ibatch = T.lscalar()
    x = T.matrix('x')
    N_x = dataset_info["N_x"]
    y = T.ivector('y')
    N_y = dataset_info["N_y"]
    g = T.nnet.sigmoid
    o = T.nnet.softmax
    ae_g = ae_o = T.nnet.sigmoid

    rng = numpy.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    initParamsFun = {"W_hl": randW, "b_hl": zerosBias, "W_ol": zerosW, "b_ol": zerosBias}
    sae = SAE( (x, N_x, y, N_y, n_hl, g, o, rng, initParamsFun), (theano_rng, ae_g, ae_o) )

    funs__pretrain = sae.compilePreTrainFunctions(x, dataset, ibatch, batch_size, pretrain_rate, corruption_levels)
    fun__finetune__train, fun__finetune__valid, fun__finetune__test = sae.compileFineTuneFunction(dataset, ibatch, batch_size, learning_rate)


    #######
    #Pre-train
    #######
    print '... pretraining'
    cost_aes   = numpy.zeros((sae.nnbp.L, N_epoch_pretrain), dtype=float)
    x_r_final  = numpy.zeros((sae.nnbp.L, dataset["train"]["N"], N_x), dtype=dataset["train"]["X"].get_value(borrow=True).dtype)
    x_c_final  = numpy.zeros((sae.nnbp.L, dataset["train"]["N"], N_x), dtype=dataset["train"]["X"].get_value(borrow=True).dtype)

    start_time = time.clock()
    for ilayer in xrange(sae.nnbp.L):
        print 'Pretrain layer %d' % ilayer
        for iepoch in xrange(N_epoch_pretrain):
            if iepoch < N_epoch-1:
                for _ibatch in xrange(n_batches["train"]):
                    cost, _, _ = funs__pretrain[ilayer](_ibatch)
                    cost_aes[ilayer][iepoch] += cost
            else:
                for _ibatch in xrange(n_batches["train"]):
                    cost, x_r_part, x_c_part = funs__pretrain[ilayer](_ibatch)
                    #x_r_final[ilayer][_ibatch*batch_size:(_ibatch+1)*batch_size] = x_r_part
                    #x_c_final[ilayer][_ibatch*batch_size:(_ibatch+1)*batch_size] = x_c_part
                    cost_aes[ilayer][iepoch] += cost
            print 'Training epoch %d, cost ' % iepoch, cost_aes[ilayer][iepoch]
    end_time = time.clock()
    pretraining_time = end_time - start_time
    print '\nPretraining complete.' + 'Ran for %.2fm' % (pretraining_time / 60.)
    aes_params_numpy = []
    for ilayer in xrange(sae.nnbp.L):
        aes_params_numpy.append (sae.aes[ilayer].W_hl.get_value(borrow=True))


    #######
    #Fine-tune
    #######
    print '... fine-tuning'
    cost_test   = numpy.zeros((N_epoch), dtype=float)
    error_valid = numpy.zeros((N_epoch/valid_epoch_frequency), dtype=int)

    start_time = time.clock()
    ivalid = 0
    iepoch = 0
    ifShouldEarlyStop = False
    while iepoch < N_epoch and ifShouldEarlyStop == False:
        for _ibatch in xrange(n_batches["train"]):
            cost_test[iepoch] += fun__finetune__train(_ibatch)

        last_error_valid = 0
        if iepoch % valid_epoch_frequency == 0:
            for _ibatch in xrange(n_batches["valid"]):
                error_valid[ivalid] += fun__finetune__valid(_ibatch)
            last_error_valid = error_valid[ivalid]
            if iepoch > 5 and ivalid > 0 and error_valid[ivalid] > error_valid[ivalid - 1] and error_valid[ivalid-1] > error_valid[ivalid - 2]:
                ifShouldEarlyStop = True
            else:
                ivalid += 1

        print 'Epoch %d complete.' % (iepoch)
        print 'Cost_test: %d, error_valid: %f%%' % (cost_test[iepoch],  float(last_error_valid)/float(n_batches["valid"]*batch_size) * 100)
        iepoch += 1
    end_time = time.clock()

    finetuning_time = end_time - start_time
    print 'Optimization complete.' + 'Ran for %.2fm' % ((finetuning_time) / 60.)


    #######
    #Test on not-seen data
    #######
    print '... testing on not-seen data'
    error_test = 0
    y_pred = numpy.array([])
    for _ibatch in xrange(n_batches["test"]):
        errors_, y_pred_ = fun__finetune__test(_ibatch)
        error_test += errors_
        y_pred = numpy.concatenate((y_pred, y_pred_), axis=0)
    print 'error_test: %f%%' % (float(error_test)/float(n_batches["test"]*batch_size) * 100)

    #######
    #Save results to files (pickling)
    #######
    print '... preparing results files :)'

    #Create info txt file
    infoFile = open(experiment_path + "/" + "info.txt", 'w')
    infoFile.write( "classification_method_name: stacked autoencoders; neural network with back-propagation, weights initialised in a greedy way with autoencoders (early stopping)\n")
    infoFile.write( prettyPrintDictionaryToString("meta_params",  meta_params  ) )
    infoFile.write("\n")
    infoFile.write( prettyPrintDictionaryToString("dataset_info", dataset_info ) )
    infoFile.write("\n")
    infoFile.write( prettyPrintDictionaryToString("learning_info", {"error_test[%]": (float(error_test))/(float(dataset["test"]["N"]))*100, "learning_time [minutes]": (finetuning_time+pretraining_time)/60., "N_epoch_actual": iepoch}) )
    infoFile.close()

    #pickle results
    pickleResultsGen(experiment_path + "/" + "infoPickled", [dataset_info, meta_params])
    nnmp_params_numpy = []
    for i in xrange(sae.nnbp.L+1):
        nnmp_params_numpy.append (sae.nnbp.params[i].get_value(borrow=True))
    results = {"cost_aes": cost_aes,
               "x_r_final": x_r_final,
               "x_c_final": x_c_final,
               "pretraining_time": pretraining_time,

               "cost_test": cost_test,
               "error_valid": error_valid,
               "error_test": error_test,
               "y_pred": y_pred,
               "finetuning_time": finetuning_time,

               "learning_time": finetuning_time+pretraining_time,
               "aes_params_numpy": aes_params_numpy,
               "nnmp_params_numpy": nnmp_params_numpy,
               "N_epoch_actual": iepoch}
    pickleResultsGen(  experiment_path + "resultsPickled",  [dataset_info, meta_params, results] )

def sae__results(experiment_path):
    dataset  = mnist_load(ifTheanoTensorShared=False)
    meta_params, results  = unpickleResultsGen(experiment_path + "/resultsPickled")
    cost           = results["cost"]
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
    #weightsAsFilters(W_hl.T, experiment_path + "/weightsAsFilters2.png")


def runExperiments():
    meta_params = {"N_epoch": 500, "N_epoch_pretrain": 50, "batch_size": 32, "learning_rate": 0.01, "pretrain_rate":0.01, "valid_epoch_frequency": 1, "n_hl": [1000, 1000, 1000],
                   "corruption_levels": [0.3, 0.35, 0.35]}
    datasetInfo = {"name": "mnist", "load_fun": mnist_load, "N_x": 28*28, "N_y": 10}
    runExperiment(meta_params=meta_params, datasetInfo=datasetInfo)

def runExperiment(meta_params, datasetInfo):
    experimentID = getTimeID()

    #Create Folder
    pathToMainDirectory = "__results/sae"
    pathToExperimentDirectory = pathToMainDirectory +'/'+ experimentID
    os.makedirs(pathToExperimentDirectory)

    #######
    #Run experiment
    #######
    sae__learn_earlystopping(pathToExperimentDirectory+"/", meta_params, datasetInfo)



if __name__ == '__main__':
    runExperiments()