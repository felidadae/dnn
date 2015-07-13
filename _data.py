
import numpy
import theano
import theano.tensor as T

import cPickle
import gzip
import os


def prepareExamplesAsShared(examples, borrow=True):
    shared_examples = theano.shared(numpy.asarray(examples, dtype=theano.config.floatX), borrow=True)
    return shared_examples

def mnist_load(ifTheanoTensorShared = True):
    """
    return dictionary:
    "train", "valid", "test"
        "X", "Y", "N"
    "X", "Y" as shared theano variable
    """
    def load_data(datasetName='mnist.pkl.gz'):
        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(datasetName)
        if data_dir == "" and not os.path.isfile(datasetName):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split('__file__')[0],
                "data",
                datasetName
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                dataset = new_path

        if (not os.path.isfile(datasetName)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, datasetName)

        #print '... loading data'

        # Load the dataset
        f = gzip.open(datasetName, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        return {"train": train_set, "valid": valid_set, "test": test_set}

    def prepareData(dataset):
        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')
        def numpy_dataset(data_xy):
            data_x, data_y = data_xy
            return data_x, data_y

        if ifTheanoTensorShared == True:
            train_set_x, train_set_y = shared_dataset(dataset["train"])
            valid_set_x, valid_set_y = shared_dataset(dataset["valid"])
            test_set_x, test_set_y   = shared_dataset(dataset["test" ])
        else:
            train_set_x, train_set_y = numpy_dataset(dataset["train"])
            valid_set_x, valid_set_y = numpy_dataset(dataset["valid"])
            test_set_x, test_set_y   = numpy_dataset(dataset["test" ])

        rval = {}
        rval["train"] = {"X": train_set_x, "Y": train_set_y, "N": dataset["train"][0].shape[0]}
        rval["valid"] = {"X": valid_set_x, "Y": valid_set_y, "N": dataset["valid"][0].shape[0]}
        rval[ "test"] = {"X":  test_set_x, "Y":  test_set_y, "N": dataset["test" ][0].shape[0]}
        return rval

    return prepareData( load_data('mnist.pkl.gz') )