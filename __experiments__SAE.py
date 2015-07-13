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
from NNBP import randW, zerosBias


def learn(path, meta_params, dataset):
    m = meta_params
    d = dataset

    rng    = m['rng']
    T_rng  = m['T_rng']
    NB = \
        {'pt'   : d['shr']['pt']   ['N'] / m['pt']['B'],
         'ft_tr': d['shr']['ft_tr']['N'] / m['ft']['B'],
         'ft_v' : d['shr']['ft_v'] ['N'] / m['ft']['B'],
         'te'   : d['shr']['te']   ['N'] / m['ft']['B']}


    #######
    #Build the model
    #######
    print '... building the model'
    ib = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    sae = SAE( (x, d['Nx'], y, d['Ny'], m['n_hl'], m['ft']['g'],
                m['ft']['o'], rng, m['ft']['initP']),
               (T_rng, m['pt']['g'], m['pt']['o']) )

    #Pretrain
    funs__pretrain = \
        sae.compile__pt(x, d['shr']['pt']['X'], ib,
        m['pt']['B'], m['pt']['K'], m['pt']['corrupt'])

    #Fine-tune
    fun__ft__tr, fun__ft__v, fun__te = \
        sae.compile__ft(d['shr'], ib, m['ft']['B'], m['ft']['K'])


    #
    ###
    #######
    #Pretraining
    #######
    ###
    #
    def print__pt_cost(ie, il, cost):
        print 'pretraining layer %d epoch %d, \n\tcost %f' % (il, ie, cost)

    print '... pretraining'
    cost_pt   = np.zeros((sae.nnbp.L, m['pt']['Ne']), dtype=float)
    #type = d['shr']['pt']['X'].get_value(borrow=True).dtype
    #x_r_final  = np.zeros((sae.nnbp.L, d['shr']['pt']['N'], d['Nx']), dtype=type)
    #x_c_final  = np.zeros((sae.nnbp.L, d['shr']['pt']['N'], d['Nx']), dtype=type)

    beg_t = time.time()

    for il in xrange(sae.nnbp.L):
        for ie in xrange(m['pt']['Ne']):
            if ie < m['pt']['Ne']-1:
                for _ib in xrange(NB['pt']):
                    cost, _, _ = funs__pretrain[il](_ib)
                    cost_pt[il][ie] += cost
            else:
                for _ib in xrange(NB['pt']):
                    cost, x_r_part, x_c_part = funs__pretrain[il](_ib)

                    #_id_beg = _ib*m['pt']['B']
                    #_id_end = (_ib+1)*m['pt']['B']
                    #x_r_final[il,_id_beg:_id_end,:] = x_r_part
                    #x_c_final[il,_id_beg:_id_end,:] = x_c_part

                    cost_pt[il][ie] += cost

            print__pt_cost(ie,il,  cost_pt[il][ie])

    end_t = time.time()
    pt_t = end_t - beg_t
    print 'Pretraining complete.' + 'Ran for %.2fm\n' % (pt_t / 60.)

    #Save params of aes
    aes_params_np = []
    for il in xrange(sae.nnbp.L):
        #We must copy arrays, because their values will change
        aes_params_np.append (sae.aes[il].W_hl.get_value(borrow=True).copy())


    #
    ###
    #######
    #Fine-tuning
    #######
    ###
    #
    def print__ft_costerror(ie, cost_tr, error_v):
        print 'finetuning epoch %d,' % ie
        print '\taverage per sample: cost_train: %f, error_valid: %f%%' % \
            (float(cost_tr)/float(d['shr']['ft_tr']['N']) * 100,
             float(error_v)/float(d['shr']['ft_v' ]['N']) * 100)

    print '... fine-tuning'
    best__nnmp_params_np  = []
    best__error_ft_v = float('Inf')
    patience_max = 10
    patience = patience_max

    cost_ft_tr = np.zeros((m['ft']['Ne']), dtype=float)
    error_ft_v = np.zeros((m['ft']['Ne']/m['ft']['VF']), dtype=int)

    beg_t = time.time()
    iv = 0
    ie = 0
    estop = False

    while ie < m['ft']['Ne'] and estop == False:
        for _ib in xrange(NB['ft_tr']):
            cost_ft_tr[ie] += fun__ft__tr(_ib)

        if ie % m['ft']['VF'] == 0:
            for _ib in xrange(NB['ft_v']):
                error_ft_v[iv] += fun__ft__v(_ib)

            if error_ft_v[iv] < best__error_ft_v:
                print 'New best model.'
                patience = patience_max
                best__error_ft_v = error_ft_v[iv]
                best__nnmp_params_np = []
                for i in xrange(len(sae.nnbp.params)):
                    best__nnmp_params_np.append(
                        sae.nnbp.params[i].get_value(borrow=True))

            else:
                if patience == 0:
                    estop = True
                    print 'Early stop set to true.'
                else:
                    patience -= 1


        print__ft_costerror(ie, cost_ft_tr[ie], error_ft_v[iv])
        ie += 1
        iv+=1

    end_t = time.time()

    ft_t = end_t - beg_t
    print 'Fine-tuning complete.' + ' Ran for %.2fm\n' % ((ft_t) / 60.)


    #
    ###
    #######
    #Testing on non-seen data
    #######
    ###
    #

    #set best model params as current ones
    for i in xrange(len(best__nnmp_params_np)):
        sae.nnbp.params[i].set_value(best__nnmp_params_np[i])

    print '... testing best model on not-seen data'
    error_te = 0
    y_pred = np.array([])
    for _ib in xrange(NB['te']):
        errors_, y_pred_ = fun__te(_ib)
        error_te += errors_
        y_pred = np.concatenate((y_pred, y_pred_), axis=0)
    error_te_perc = float(error_te)/float(d['shr']['te']['N']) * 100
    print 'error_test: %f%%\n' % (error_te_perc)


    #######
    #Save results to files (pickling)
    #######
    print '... preparing results files :)'

    #Create info txt file

    def createInfoFile(experiment_path, meta_params, dataset_info,
        error_test, dataset, finetuning_time, pretraining_time, iepoch):
        #Create info txt file
        infoFile = open(experiment_path + '/' + 'info.txt', 'w')
        infoFile.write(
            'classification_method_name: stacked autoencoders; \
            \n\tneural network with back-propagation, weights initialised in a \
            \n\tgreedy way with autoencoders (early stopping).\n')
        infoFile.write( prettyPrintDictionaryToString('meta_params',  meta_params  ) )
        infoFile.write('\n')
        infoFile.write( prettyPrintDictionaryToString('dataset_info', dataset_info ) )
        infoFile.write('\n')
        error_test_percent = (float(error_test))/(float(dataset['te']['N']))*100
        learning_time_minutes = (finetuning_time+pretraining_time)/60.
        infoFile.write( prettyPrintDictionaryToString('learning_info',
            {'error_test[%]': error_test_percent,
             'learning_time [minutes]': learning_time_minutes,
             'N_epoch_actual': iepoch}) )
        infoFile.close()

    createInfoFile(path, m, d, error_te, d['shr'], ft_t, pt_t, ie)

    #pickle results
    def pickleThatResults(
        experiment_path, meta_params,
        error_test, finetuning_time, pretraining_time, iepoch,
        cost_aes,
        #x_r_final, x_c_final,
        cost_test, error_valid, y_pred,
        aes_params_np, nnmp_params_np):

        pickleResultsGen(experiment_path + '/' + 'infoPickled', [meta_params])

        #######
        #RESULTS Dictionary
        ######$
        results = \
            {'cost_aes': cost_aes,
            #'x_r_final': x_r_final,
            #'x_c_final': x_c_final,
            'aes_params_np': aes_params_np,
            'pretraining_time': pretraining_time,
            ##
            'cost_test': cost_test,
            'error_valid': error_valid,
            'error_test': error_test,
            'y_pred': y_pred,
            'nnmp_params_np': nnmp_params_np,
            'finetuning_time': finetuning_time,
            ##
            'learning_time': finetuning_time+pretraining_time,
            'N_epoch_actual': iepoch}

        pickleResultsGen(experiment_path + 'resultsPickled',
            [meta_params, results] )

    pickleThatResults(
        path, m,
        error_te, ft_t, pt_t, ie,
        cost_pt, #x_r_final, x_c_final,
        cost_ft_tr, error_ft_v, y_pred,
        aes_params_np, best__nnmp_params_np)


def run(meta_params, dataset_info, folderName=''):
    experimentID = getTimeID()

    #Create Folder
    if folderName != '':
        folderName = '/'+folderName
    pathToMainDirectory = '__results/sae' + folderName
    pathToExperimentDirectory = pathToMainDirectory +'/'+ experimentID
    os.makedirs(pathToExperimentDirectory)

    #Run
    learn(pathToExperimentDirectory+'/', meta_params, dataset_info)

def analise(experiment_path):
    dataset  = mnist_load(ifTheanoTensorShared=False)
    meta_params, results  = unpickleResultsGen(experiment_path + '/resultsPickled')
    cost           = results['cost']
    x_r_final       = results['x_r_final']
    ae_params_np = results['ae_params_np']
    W_hl,b_hl = ae_params_np

    #PrintDescription (name of dataset)(N_train, N_valid, N_test)(N_epoch, batch_size, learning_rate, valid_epoch_frequency)
    seperateLine(before=True)
    print 'Method name: auto-encoder'
    print 'Dataset: MNIST'
    print 'Meta params:'
    prettyPrintDictionary(meta_params)
    seperateLine(after=True)

    #Learning plot: cost(ibatch)
    costPlot(cost, experiment_path + '/costplot.pdf')

    #Show predicted images
    mnist_visualise ( dataset['tr']['X'],    (0, 500), 'ORIGINAL',       experiment_path + '/originalSet.png'        )
    mnist_visualise ( results['x_c_final' ],    (0, 500), 'CORRUPTED',      experiment_path + '/corruptedSet.png'       )
    mnist_visualise ( x_r_final,                (0, 500), 'RECONSTRUCTED',  experiment_path + '/reconstructedSet.png'   )

    #Filters from weightse
    mnist_visualise(  W_hl.T, (0,meta_params['N_hl']), 'Weights as filters (W_hl.T)', experiment_path + '/weightsAsFilters.png')
    weightsAsFilters(W_hl.T, experiment_path + '/weightsAsFilters2.png')


def performExperiment__pretrainInfluence():
    ###########
    #Load data
    ###########
    original_mnist_shared = mnist_load()
    nist_letters = unpickleResultsGen("__utils/smallHappyLetters")
    nist_letters = nist_letters/255.
    nist_letters = nist_letters[0:50000,:]
    pretrain_train = nist_letters
    #    np.concatenate(
    #        (original_mnist_shared['train']['X'].get_value(borrow=True).copy(),
    #         nist_letters), 0 )
    np.random.shuffle(pretrain_train)
    pretrain_train_shared = \
        {'X': prepareExamplesAsShared(pretrain_train),
        'N':  pretrain_train.shape[0]}


    ###########
    #Random number generator
    ###########
    rng = np.random.RandomState(9211)
    T_rng = RandomStreams(rng.randint(2 ** 12+5))


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
            'Ne': 500,
            'initP': initP,
            'B': 1,
            'K': 0.01,
            'VF': 1},
        'rng': rng,
        'T_rng': T_rng}


    dataset_info = \
        {'shr':
             {'pt':     original_mnist_shared['train'],
              'ft_tr':  original_mnist_shared['train'],
              'ft_v' :  original_mnist_shared['valid'],
              'te':     original_mnist_shared['test' ]},
         'Nx': 784,
         'Ny': 10}


    #
    ###
    #######
    #run
    #######
    ###
    #
    #folderName = '500_500_500__'
    #meta_params['pt']['Ne'] = 0
    #for i in xrange(10):
    #    run(meta_params, dataset_info, folderName)

    #folderName = '900_500_500__04_035_035__20__pretrain_standard'
    #meta_params['n_hl'] = [900,500,500]
    #meta_params['pt']['corrupt'] = [0.4, 0.35, 0.35]
    #meta_params['pt']['Ne'] = 20
    #meta_params['ft']['Ne'] = 1000
    #for i in xrange(10):
    #    run(meta_params, dataset_info, folderName)

    #folderName = '500_500_500_pretrain_onlyletters'
    #meta_params['pt']['Ne'] = 15
    #meta_params['ft']['Ne'] = 1000
    #dataset_info['shr']['pt'] = pretrain_train_shared
    #for i in xrange(8):
    #    run(meta_params, dataset_info, folderName)

    folderName = '500_'
    meta_params['pt']['Ne'] = 0
    for i in xrange(10):
        run(meta_params, dataset_info, folderName)










if __name__ == '__main__':
    performExperiment__pretrainInfluence()
    #analise('__results/sae/distraction_influence_01/25-05-2015__23-48-36')