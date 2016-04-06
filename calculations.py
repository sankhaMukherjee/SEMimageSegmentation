import numpy as np 
from scipy.spatial import ConvexHull


def pointInPatch(x,y,poly):
    '''
        This function takes a point x,y and determines if it 
        lies inside the polygons determined by the vertices of
        the patch. We do 2 levels of checks. In the first level
        we see if a point is outside the min/max values of the 
        patch. Then and only then do we proceed for the next test.
        
        The patch is a list that looks like so: [(float, float)]
    '''

    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def polygonIntersection(poly1, poly2):
    '''
        If two polygons intersect, this function returns 
        True else returns False
    '''

    poly1 = zip(*poly1)
    poly2 = zip(*poly2)

    for (x,y) in poly1:
        if pointInPatch(x,y,poly2): return True

    return False

def findIntersections(polygons, skipLength=40, verbose=False):
    '''
        Given a list of polygons, this functions maps the 
        polygons which are close to each other ...
    '''

    plNos = range(len(polygons))

    if verbose: print 'Total polygons: ', len(polygons)

    allData = []

    for i in plNos:
        # if verbose: print
        if verbose: print i,
        temp = []
        for j in plNos:
            
            if abs(i-j) > skipLength: 
                temp.append(False)
            else:
                if i == j:
                    temp.append(False)
                else:
                    temp.append( polygonIntersection(polygons[i], polygons[j]) )

        allData.append(temp)

        if verbose: print

    return allData

def calculateArea(x, y):
    '''
        Given the x and y coordiantes of a convex polygon, this 
        function finds the area of the polygon. The area of the 
        polygon is defined in the following equation:


        A = 0.5 * ( x1.y2  - x2.y1 + x2.y3 - x3.y2 + ... + xn.y1 - x1.yn)

        This may be simplified to:

        A = 0.5 * ( x1.y2 + x2.y3 + ... + xn.y1)
          - 0.5 * ( x2.y1 + x3.y2 + ... + x1.yn)

        These operations can be vectorized ...

        Reference:
        1. http://mathworld.wolfram.com/PolygonArea.html
    '''

    x = np.array(x)
    y = np.array(y)

    v1 = (x[:-1] * y[1:]).sum()
    v2 = (x[1:]  * y[:-1]).sum()

    return np.abs(0.5 * (v1 - v2))

def centerPoint(x, y):
    '''
        This function returns the center point of the 
        convex polygon provided ...
    '''
    xMean = np.array(x).mean()
    yMean = np.array(y).mean()

    return xMean, yMean

def deleteEdges(allData, allXY):
    yMax, xMax = np.shape(allData['newImage'])
    
    print 'Deleting Edge polygons ...'
    newXY = []
    for (x,y) in allXY:
        if 0 in x: continue
        if 0 in y: continue
        if (xMax-1) in x: continue
        if (yMax-1) in y: continue

        newXY.append( (x,y) )

    return newXY


