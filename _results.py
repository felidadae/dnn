
import cPickle
import numpy
import matplotlib.pyplot as plt
import pylab

from scipy.interpolate import interp1d

import PIL.Image as Image
from _utils import printmat, tile_raster_images


def plotConfusionMatrix(x, N, y, y_pred, pathToFile):
    y_unique = list(set(y))
    def prepareMatrixOfPrediction(y_unique, y, y_pred):
        D =  len( y_unique )
        m = numpy.zeros((D,D), dtype=int)
        for iy in xrange(len(y)):
            idx_iy       = y_unique.index(y     [iy])
            idx_iy_pred  = y_unique.index(y_pred[iy])
            m[idx_iy, idx_iy_pred] += 1
        return m

    conf_arr = prepareMatrixOfPrediction(y_unique, y, y_pred)
    import cmath
    plt.rcParams['font.family'] = "Arial"
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(numpy.array(norm_conf), cmap=plt.cm.Blues,
                    interpolation='nearest')

    width  = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            if x == y :
                ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', color='white')
            else:
                ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', color='black')

    #cb = fig.colorbar(res)
    plt.xticks(y_unique)
    plt.yticks(y_unique)
    plt.title("Confusion matrix")
    ax.set_aspect(1)
    plt.savefig(pathToFile, format='pdf')
    plt.close()

def matrixOfConfusions (x, N, y, y_pred):
    y_unique = list(set(y))
    def prepareMatrixOfPrediction(y_unique, y, y_pred):
        D =  len( y_unique )
        m = numpy.zeros((D,D), dtype=int)
        for iy in xrange(len(y)):
            idx_iy       = y_unique.index(y     [iy])
            idx_iy_pred  = y_unique.index(y_pred[iy])
            m[idx_iy, idx_iy_pred] += 1
        return m

    matrixOfPrediction = prepareMatrixOfPrediction(y_unique, y, y_pred)
    print 'Matrix of confusions (vertical -> y, horizontal -> y_pred):'
    printmat(matrixOfPrediction, row_labels=y_unique, col_labels=y_unique)

def testCost_validError_relationPlot (costs_test, errors_valid):
        N_epoch = costs_test.shape[0]
        N_valid_frequency = costs_test.shape[0] / errors_valid.shape[0]
        x =  numpy.array( range(0, N_epoch) )
        y1 = costs_test
        y2 = errors_valid

        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.title('Cost/Error(iepoch)')
        plt.ylabel('Cost/Error function')
        plt.xlabel('epoch index')
        plt.show()

def costPlot(costs, pathToFile = "",xtitle="epoch index"):
    xMax = costs.shape[0]
    x =  numpy.array( range(0, xMax), dtype = int )
    y1 = costs
    xnew = numpy.linspace(0, xMax-1, (xMax-1)*20)
    f2 = interp1d(x, y1, kind='cubic')

    if xMax < 30:
        plt.plot(xnew, f2(xnew), '-g', alpha = 0.5)
    plt.scatter(x,y1, s=30, marker='o', edgecolor='black', linewidth='0', facecolor='green',alpha=0.8)
    #plt.plot(x, y1, 'og')
    if xMax < 20:
        plt.xticks(x)
    plt.title('Cost function plot')
    plt.ylabel('cost')
    plt.xlabel(xtitle)
    plt.xlabel('epoch index')
    plt.savefig(pathToFile)
    plt.close()

def mnist_visualise(examples, rangeOfExamples, stringIDOfSet, pathToFile = ""):
    def example(idx):
        return examples[idx,:].reshape(28,28)

    def visualizeExamples(min, n, rowLength):
        Ncol = rowLength
        Nrow = n/rowLength
        for irow in xrange(Nrow):
            for icol  in xrange(Ncol):
                if icol == 0:
                    row = example(min + irow*Ncol+icol)
                else:
                    row = numpy.concatenate((row, example(min + irow*Ncol+icol)), axis = 1)
            if irow == 0:
                result = row
            else:
                result = numpy.concatenate((result, row), axis = 0)
        pylab.imshow(result, cmap=pylab.cm.gray)
        title = "Examples in range: (" + min.__str__()+" - " + (n+min).__str__() + ") of " + stringIDOfSet + " set"
        #print(title)
        pylab.title(title)
        pylab.xlabel('')
        pylab.ylabel('')
        pylab.xticks([])
        pylab.yticks([])
        pylab.savefig(pathToFile, dpi=220)
    l,r = rangeOfExamples
    visualizeExamples( l, r, 25 )

def weightsAsFilters(WT, pathToFile, filter_dim = (28,28), tile_y = 25):
    N_neurons = WT.shape[0]
    image = Image.fromarray(tile_raster_images(X = WT, img_shape=filter_dim, tile_shape=(N_neurons/tile_y, tile_y), tile_spacing=(1, 1)))
    image.save(pathToFile)
    image.show()



def pickleResults(fileName, meta_params, costs_test, errors_valid, error_test, y_pred__test, model_params):
    save_file = open('__results/pickling/' + fileName, 'wb')  # this will overwrite current contents
    cPickle.dump(meta_params, save_file, -1)
    cPickle.dump(costs_test, save_file, -1)
    cPickle.dump(errors_valid, save_file, -1)
    cPickle.dump(error_test, save_file, -1)
    cPickle.dump(y_pred__test, save_file, -1)
    #cPickle.dump(u.get_value(borrow=True), save_file, -1)
    save_file.close()

def unpickleResults(fileName):
    save_file = open('__results/pickling/' + fileName)
    meta_params  = (cPickle.load(save_file))
    costs_test   = (cPickle.load(save_file))
    errors_valid = (cPickle.load(save_file))
    error_test   = (cPickle.load(save_file))
    y_pred__test = (cPickle.load(save_file))
    model_params = []
    return (meta_params, costs_test, errors_valid, error_test, y_pred__test, model_params)

def pickleResultsGen(fileName, listOfVars):
    save_file = open(fileName, 'wb')  # this will overwrite current contents
    cPickle.dump(listOfVars, save_file, -1)
    save_file.close()

def unpickleResultsGen(fileName):
    save_file = open(fileName)
    listOfVars  = (cPickle.load(save_file))
    return listOfVars