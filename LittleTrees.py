__author__ = 'Kadlubek47'

import PIL.Image
import PIL.ImageOps
import numpy




def PIL2array(img):
    return numpy.array(img.getdata(),
                    numpy.float32).reshape(img.size[1], img.size[0])
                    
def showSmallSadTree(trees, itree, D_tree):
    imgPIL = PIL.Image.frombuffer("F", (D_tree[0],D_tree[1]), trees[itree])
    imgPIL.show()

def readImageIntoNumpyArray(path):
    im = PIL.Image.open(path)
    im = im.convert("F")
    #print im.format, im.size, im.mode
    #print im.size[0]/28, im.size[1]/28, im.size[0]/28 * im.size[1]/28

    numpy_im = PIL2array(im)
    numpy_im.shape
    
    return numpy_im
    
def prepareSmallTrees(path, D_tree, N_tree):
    img = readImageIntoNumpyArray(path)
    
    rng = numpy.random.RandomState(123)
    x_trees = rng.randint(0, img.shape[0]-D_tree[0], N_tree)
    y_trees = rng.randint(0, img.shape[1]-D_tree[1], N_tree)
    L_tree = D_tree[0] * D_tree[1]
    
    trees = numpy.zeros(shape=(N_tree, L_tree), dtype=numpy.float32)
    for itree in xrange(N_tree):
        x_tree = x_trees[itree]
        y_tree = y_trees[itree]
        trees[itree] = img[ x_tree:x_tree+D_tree[0] ][:, y_tree:y_tree+D_tree[1]].reshape(L_tree)
    return trees
    
def prepareMNISTTrainSet(N = 50000):
    return prepareSmallTrees("pacz.jpg", [28,28], N)
    

    
    
    
mnist_visualise ( dataset["train"]["X_distracted"].get_value(borrow=True), (0, 500), "ORIGINAL", "dupa.png")
mnist_visualise ( x_r_final, (0, 500), "ORIGINAL", "dupa.png")


ilayer = 0
print 'Pretrain layer %d' % ilayer
for iepoch in xrange(5):
    for _ibatch in xrange(n_batches["train"]):
        cost, x_r_part, x_c_part = funs__pretrain[ilayer](_ibatch)
        x_r_final[ilayer,_ibatch*batch_size:(_ibatch+1)*batch_size,:] = x_r_part
        x_c_final[ilayer,_ibatch*batch_size:(_ibatch+1)*batch_size,:] = x_c_part
        cost_aes[ilayer][iepoch] += cost
    print 'Training epoch %d, cost ' % iepoch, cost_aes[ilayer][iepoch]