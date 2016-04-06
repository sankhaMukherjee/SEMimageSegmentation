import cv2
import pylab as pl 
import numpy as np 
import random
from scipy              import ndimage as ndi
from scipy.spatial      import ConvexHull
from skimage            import exposure
from skimage.feature    import peak_local_max
from skimage.filters    import threshold_adaptive
from skimage.morphology import erosion, dilation, watershed


def equalize(data, clip_limit=0.03):
    return exposure.equalize_adapthist(data, clip_limit=clip_limit)

def LPF(data):
    # Low pass filtering
    kernel = np.array([
                        [1,1,1,1,1],
                        [1,2,2,2,1],
                        [1,2,2,2,1],
                        [1,2,2,2,1],
                        [1,1,1,1,1]  ]) / 34.0

    return cv2.filter2D(data, -1, kernel)

def threshold(data, block_size = 41):

    return threshold_adaptive(data, block_size)

def erode(data, erodrN = 2):
    
    eroded = data.copy()
    for i in range(erodrN):
        eroded = erosion(eroded)

    return eroded

def segmentize(eroded, dist=30):

    distance   = ndi.distance_transform_edt(eroded)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((dist, dist)), labels=eroded)
    markers    = ndi.label(local_maxi)[0]
    labels     = watershed(-distance, markers, mask=eroded)

    # Now randomize the lables so the lables are 
    # easy to see on the image ...
    initLabels = [l for l in list(set(list(labels.ravel()))) if l > 0]
    diffLabels = [l for l in list(set(list(labels.ravel()))) if l > 0]
    # random.shuffle(diffLabels)

    newimage = np.zeros(np.shape(labels))

    for i, d in zip(initLabels, diffLabels):
        newimage[labels==i] = d

    return initLabels, eroded, labels, newimage

def plotting(allData, allXY):

    print 'Generating image data ...'

    pl.figure(figsize=(9,6))

    pos1 = 1.0/3.0
    pos2 = 2*pos1

    ax = []
    for i in range(3):
        ax.append(pl.axes([i*pos1, 0.5,   pos1, 0.5]))
        ax.append(pl.axes([i*pos1, 0.0, pos1, 0.5]))

    locs = {
        'original'  : 0,
        'equalized' : 2,
        'LPF'       : 4,
        'threshold' : 1,
        'erosion'   : 3}

    for l in locs.keys():
        ax[ locs[l] ].imshow(allData[l], 
                                cmap=pl.cm.gray)#, aspect='auto')
        ax[ locs[l] ].set_xticks([])
        ax[ locs[l] ].set_yticks([])
        ax[ locs[l] ].set_title( l )


    ax[ -1 ].imshow(allData['original'], cmap=pl.cm.gray)
    for (X,Y) in allXY:
        pl.plot(X, Y, 'indianred', lw=1)
    
    yMax, xMax = np.shape(allData['newImage'])

    ax[ -1 ].set_xticks([])
    ax[ -1 ].set_yticks([])
    ax[ -1 ].set_xlim([0, xMax])
    ax[ -1 ].set_ylim([0, yMax])
    ax[ -1 ].set_title( 'segmentation' )

    return

def plotFinalImage(data, allXY, lineColor='salmon'):

    yMax, xMax = np.shape(data)
    
    pl.figure(figsize=(6,6.0*yMax/xMax))
    ax = pl.axes([0,0,1,1])
    pl.imshow(data, cmap=pl.cm.gray)
    for (X,Y) in allXY:
        pl.plot(X, Y, color=lineColor)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, xMax])
    ax.set_ylim([0, yMax])

    return

def processImage(data, config):

    print 'Separating the image into data vs. info ...'
    # Separating the image into data vs. info ...
    infoStart = np.where(data[:,:10].mean(axis=1)==0)[0][0]
    dataInfo  = data[infoStart:,:]
    data      = data[:infoStart,:]

    print 'Adaptive equalization ...'
    # Adaptive Equalization
    clip_limit = config['equalize']['clip_limit']
    data_adapteq = equalize(data.copy(), clip_limit=clip_limit)

    print 'Low Pass filter ...'
    # Low pass filter
    smooth = LPF(data_adapteq.copy())

    print 'Adaptive Thresholding ...'
    # Adaptive thresholding ...
    block_size = config['threshold']['block_size']
    binary_adaptive = threshold(smooth.copy(), block_size=block_size)
    
    print 'Erosion ...'
    # Erosion 
    erodrN = config['erode']['N']
    eroded = erode(binary_adaptive.copy()==False, erodrN = erodrN)
    eroded = eroded == False
    
    print 'Segmentation and distance metric calculations ...'
    # Distance metric ...
    dist = config['dist']['dist']
    initLabels, distance, labels, newImage = segmentize(eroded.copy(), dist=dist)

    allData = {
        'original'  : data,
        'equalized' : data_adapteq,
        'LPF'       : smooth,
        'threshold' : binary_adaptive,
        'erosion'   : eroded,
        'newImage'  : newImage,
        'initLabels': initLabels,
    }

    return allData

def findXY(newImage, i):
    '''
        Given the newImage and the ith label, this function is 
        going to return the vertices of the convex hull of the 
        set of points defining the image.

        Remember that these are closed polygons. i.e.

        x = [x0, x1, ..., xn, x0]
        y = [y0, y1, ..., yn, y0]
    '''

    points = np.where( newImage == i)
    points = np.array(points).T
    hull   = ConvexHull(points)
    verts1 = list(hull.vertices)
    verts1 = verts1 + [verts1[0]]
    
    x = points[verts1, 1]
    y = points[verts1, 0]
    
    return x, y


def polygonIntersections():
    print 'Finding polygon intersections ...'
    skipLength = config['skipLength']
    cross = calc.findIntersections(newXY, skipLength=skipLength, verbose=False)
    cross = np.array(cross).astype(int)
    cross = np.tril( cross + cross.T )

    # Deal with the Intersections ...
    # This doen only one level of intersections ...
    cX, cY = np.where(cross)

    return cX, cY


def findIntersectionGroups(cX, cY):
    '''
        This uses the 'old' algorithm. This is not very efficient because
        this causes the same polygon to be part of more than one group. This 
        is not ideal ...
    '''
    intersectionGroups = []

    for c in sorted(zip(cX, cY), key=min):
        # if c exists in a particular group, insert it there
        joined = False
        for i in range(len(intersectionGroups)):
            if any(c1[0] in intersectionGroups[i]):
                intersectionGroups[i] += list(c)
        
        # If this doesnt happen, start a new one ...
        if not joined:
            intersectionGroups.append(list(c))

    # Remove duplicates ...
    intersectionGroups = [sorted(list(set(i))) for i in intersectionGroups]
    intersectionGroups = sorted(list(map(eval,set(map(str, intersectionGroups)))), key=min)

    return intersectionGroups


