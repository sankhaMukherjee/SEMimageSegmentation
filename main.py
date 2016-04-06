import image as im 
import pylab as pl 
import numpy as np
import calculations as calc
from scipy.spatial      import ConvexHull

def findNNdistances(centers):

    centers = np.array(centers)
    nearestDist = []
    for c in centers:
        nearestC = sorted(centers - c, key= lambda m: sum(np.abs(m)) )[1]
        nearestDist.append( np.linalg.norm(nearestC) )

    return nearestDist


if __name__ == '__main__':

    pl.ion()

    with open('config.cfg') as f: 
        config = eval(''.join(f.readlines()))
    print config

    fileName = config['fileName']

    print 'Reading the image'
    with open(fileName) as f: 
        data = np.array([map(float, f.strip('\n').split()) for f in f.readlines()])

    data = data/data.max()

    allData  = im.processImage(data, config)
    allXY    = []
    allAreas = []

    print 'Computing the vertices of the polygons ...'
    for i in allData['initLabels']:
        allXY.append( im.findXY(allData['newImage'], i) )

    print 'Deleting the edges of the polygons ...'
    newXY = calc.deleteEdges(allData, allXY)

    print 'Computing the areas of the polygons ...'
    for (x,y) in newXY:
        allAreas.append( calc.calculateArea(x,y) )


    print 'Computing the centers of the polygons ...'
    centers = []
    for (x,y) in newXY:
        centers.append( calc.centerPoint(x,y) )

    print 'Finding the nearest neighbour distance of the polygon centers ...'
    dist = findNNdistances(centers)

    scale    = float(config['physicalLen'])/config['imageLen']
    dist     = np.array(dist)*scale
    allAreas = np.array(allAreas)*scale

    im.plotting(allData, newXY)
    if config['saveImage']: pl.savefig(config['fileName']+'_Overall.png', dpi=300)

    im.plotFinalImage(data, newXY)
    if config['saveImage']: pl.savefig(config['fileName']+'_FinalImage.png', dpi=300)
    
    xC, yC = zip(*centers)
    pl.plot(xC, yC, '+')

    pl.figure(figsize=(4,3))
    pl.axes([0.09, 0.17, 0.93-0.09, 0.95-0.17])
    n, bins, patches = pl.hist(allAreas, bins=60)
    for p in patches: pl.setp( p, color='indianRed' )
    pl.xlabel('Areas (nm$^2$)')
    if config['saveImage']: pl.savefig(config['fileName']+'_AreaHist.png', dpi=300)
    
    pl.figure(figsize=(4,3))
    pl.axes([0.09, 0.17, 0.96-0.09, 0.93-0.17])
    n, bins, patches = pl.hist(dist, bins=60)
    for p in patches: pl.setp( p, color='indianRed' )
    pl.xlabel('Center Center distances (nm)')
    if config['saveImage']: pl.savefig(config['fileName']+'_DistHist.png', dpi=300)

    pl.show()