
import numpy
import PIL.Image
import PIL.ImageOps



def format__1(digits,num):
        if digits<len(str(num)):
            raise Exception("digits<len(str(num))")
        return ' '*(digits-len(str(num))) + str(num)

def printmat(arr,row_labels=[], col_labels=[]): #print a 2d numpy array (maybe) or nested list
    max_chars = 5 #the maximum number of chars required to display any item in list
    if row_labels==[] and col_labels==[]:
        for row in arr:
            print '[%s]' %(' '.join(format__1(max_chars,i) for i in row))
    elif row_labels!=[] and col_labels!=[]:
        rw = max([len(str(item)) for item in row_labels]) #max char width of row__labels
        print '%s %s' % (' '*(rw+1), ' '.join(format__1(max_chars,i) for i in col_labels))
        for row_label, row in zip(row_labels, arr):
            print '%s [%s]' % (format__1(rw,row_label), ' '.join(format__1(max_chars,i) for i in row))
    else:
        raise Exception("This case is not implemented...either both row_labels and col_labels must be given or neither.")

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar



def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array



def seperateLine(before = False, after = False):
        if before == True:
            print '\n'
        print ''
        if after == True:
            print '\n'

def prettyPrintDictionary(dict, tabsNum = 0):
    keys = dict.keys()
    values = dict.values()
    N = len(keys)
    for i in xrange(N):
        print "\t", keys[i], ": ", values[i]

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

def printDictionaryAsOneString(dict, seperatorOfParis = "__"):
    keys = dict.keys()
    values = dict.values()
    N = len(keys)
    sums = str
    for i in xrange(N):
        sums+= keys[i], ": ", values[i]

def getTimeID():
    import time
    return time.strftime("%d-%m-%Y")+'__'+time.strftime("%H-%M-%S")



def PIL2array(img):
    numpy_img =  numpy.array(img.getdata(),
                    numpy.float32).reshape(img.size[1], img.size[0])
    numpy_img = numpy_img/255.0
    return numpy_img

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

def mixTwoSets(base, distractor, K_distract):
    N = base.shape[0]
    N_distract = int(K_distract * N)

    mix = base.copy()

    import random
    random_list = random.sample(xrange(N), N_distract)

    idistractor = 0
    for imix in xrange (N_distract):
        mix[ random_list[imix] ] = distractor[random_list[imix]]
        idistractor += 1

    return mix