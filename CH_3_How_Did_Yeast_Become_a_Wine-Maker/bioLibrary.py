__author__ = 'ilap'

# Import modules and setup
import math

import matplotlib.pyplot as plt
'''plt.ion ()
from easyplot import EasyPlot'''

import threading
import numpy as np
import sys
from copy import *
import sys
sys.setrecursionlimit(1000000000)
import time


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

'''
Matrix class
'''

class Matrix:

    def __init__(self, string_list="", n=0, m=0, dtype=np.int16):

        self.row_len = n
        self.dimension = m # same as col_len

        print self.row_len, self.dimension
        if self.row_len:
            if not self.dimension:
                self.dimension = self.row_len
            arr = [0]*self.row_len*self.dimension
        else:
            if not string_list:
                print "ERROR", exit ()
            else:
                arr = string_list.split ()
                arr_len = len (arr)


            if self.dimension:
                self.row_len = arr_len/self.dimension
            elif arr_len:
                self.row_len = int (math.sqrt (arr_len))
                self.dimension = self.row_len
            else:
                print "ERROR", exit ()

            print "ARRAY", self.row_len, self.dimension, arr
        self.matrix = []
        self.matrix = np.array (arr, dtype=float) #fromiter (arr)

        #self.matrix = np.around (self.matrix, decimals=2)

        self.matrix = np.reshape(self.matrix, (self.row_len, self.dimension))

    def getMatrix(self):
        return self.matrix



class ClusterMatrix (Matrix):
    def __init__(self, string_list="", n=0, m=0, dtype=np.int16):
        Matrix.__init__(self, string_list, n, m, dtype)
        self.labels = []

    '''
    d(v, w) = \sqrt{\sum_{j=1}^m  (v_j - w_j)^2}

    '''
    def eucledianDistance (self, v, w):
        distance = 0.0
        for j in range (self.dimension):
            distance += (v[j]-w[j]) ** 2

        return round (math.sqrt (distance), 4)

    '''
      d(data_point, centers) = min all points x from Centers distance (data_point, center or x)
    '''
    def minDataPointDistanceToCenters (self, data_point, centers):
        #print "minDPDTC", data_point, centers

        distance = 0
        distance = min ([self.eucledianDistance(data_point, center) for center in centers])
        return distance
    '''
        maxDistance (data_points, centers) = max all points data_point from data_points distance(data_point, centers)
    '''
    def maxDataPointsDistanceToCenters (self, data_points, centers):
        print "minDPDTC", data_points, centers

        data_point = []
        max_val = -1

        for (idx, dp) in enumerate (data_points):
            print "IDX", idx, dp
            d = self.minDataPointDistanceToCenters(dp, centers)

            if d > max_val:
                max_val = d
                data_point = dp
                res_idx = idx
        return (data_point, max_val, res_idx)


    def farthestFirstTraversal (self, k):

        data_point = self.getMatrix()[0]

        centers = [data_point]
        while len (centers) < k:

            (data_point, distance, idx) = self.maxDataPointsDistanceToCenters(self.getMatrix(), centers)
            centers.append(data_point)

        print centers
        return centers

    def testsquaredErrorDistorttion (self, data_points, centers):
        distortion = 0
        distortion = sum ([(self.minDataPointDistanceToCenters(data_point, centers)**2) for data_point in data_points])/len (data_points)
        return distortion

    def squaredErrorDistorttion (self,  data_points, centers):
        distortion = 0
        distortion = sum ([(self.minDataPointDistanceToCenters(data_point, centers)**2) for data_point in data_points])/len (data_points)
        return (data_point, distortion)

    """
        Get Center of Gravity
    """
    def getCoG (self, data_points):
        m = self.dimension
        n = len (data_points)

        result = [0.0]*m
        for i in range (m):
            v = sum (data_points[:,i]) / n
            #v = round (v, 4)
            #if (v-x) < 0:
            #    v -= 0.0001
            #print v, x
            result[i] = v

        #print "TUPLE", result
        #exit ()
        return tuple (result)

    def getClosestCenter (self, data_point, centers):
        distance = 0
        min_val = sys.maxint

        #distortion = sum ([(self.minDataPointDistanceToCenters(data_point, centers)**2) for data_point in data_points])/len (data_points)


        for center in centers:
            if (data_point == center):
                continue
            d = self.eucledianDistance(data_point, center)
            if d < min_val:
                min_val = d
                chosen_center = center

        return (chosen_center, min_val)

    def indexOfCenter (self, center, centers):

        result = -1
        for idx, c in enumerate (centers):
            if (c == center).all ():
                return idx

        return result


    def isIn (self, c, cs):
        result = False
        for i in cs:
            if (c == i):
                return True
        return result

    def centersToCluster (self):
        pass

    def clusterToCenters (self):
        pass

    def kmeansClusteringGood (self, k):

        """return self.getCoG (self.getMatrix())"""

        PoC =  {}
        centers = self.getMatrix()[0:k]
        for center in centers:
            #print tuple(map(tuple, center))
            PoC [(center[0], center[1])] = []
        print "POC KEYS", PoC.keys()

        data_points = self.getMatrix()

        while True:

            for data_point in data_points:
                if self.isIn( data_point, centers):
                    continue

                (c, d) = self.getClosestCenter (data_point, centers)

                ct = (c[0], c[1])
                #pt = (data_point[0], data_point[1])
                pt = data_point


                if PoC.has_key(ct):
                    PoC[ct].append (pt)
                else:
                    PoC[ct] = [pt]

                '''print "WWWWWWW", PoC
                print "DP, CENTER, DIST", data_point, c, d'''

            new_PoC = {}
            nc = []

            print "POC KEYS2", PoC.keys()
            for center in PoC.keys():
                #print "CENTER", center
                if PoC[center] != []:
                    arr = np.array (PoC[center])
                    #print arr
                    CoG = self.getCoG(arr)
                    print "NOTEMPTY", center, CoG
                    new_PoC[CoG] = []
                    nc.append (CoG)
                else:
                    new_PoC[center] = []
                    nc.append (center)

            print "NC", nc
            new_centers = np.array (nc)

            print
            print "Old Centers", ' '.join (str (x) for x in centers.tolist ())
            print "New Centers", ' '.join (str (x) for x in new_centers.tolist ())

            '''
            print "Points1", PoC.keys()[0],"###: ", ' '.join (str (x) for x in PoC[PoC.keys()[0]])
            print "Points2", PoC.keys()[1],"###: ",  ' '.join (str (x) for x in PoC[PoC.keys()[1]])
            #print "Points3", PoC.keys()[2],"###: ",  ' '.join (str (x) for x in PoC[PoC.keys()[2]])'''

            print centers == new_centers
            print "CCC", (centers)
            print "EEE", (new_centers)
            if (centers == new_centers).all ():
                print "Finished##########################"
                print nc
                break
            else:
                PoC = new_PoC
                centers = new_centers

    def kmeansClustering (self, k):

        """return self.getCoG (self.getMatrix())"""

        PoC =  {}

        centers = [ a.tolist () for a in self.getMatrix()[0:k]]
        for center in centers:
            PoC [tuple (center)] = []

        data_points = [ a.tolist () for a in self.getMatrix()]

        while True:
            (aa, dist) = self.squaredErrorDistorttion (data_points, centers)
            '''print "Centers.....", dist
            for r in centers:
                print ' '.join ( str ("%0.4f" % x) for x in r)'''

            for data_point in data_points:

                if data_point in centers:
                    continue

                (c, d) = self.getClosestCenter (data_point, centers)

                ct = tuple (c)
                pt = data_point

                if PoC.has_key(ct):
                    PoC[ct].append (pt)
                else:
                    PoC[ct] = [pt]

            new_PoC = {}
            new_centers = []

            #print "POC KEYS2", PoC.keys()
            for center in PoC.keys():
                if PoC[center] != []:
                    arr = np.array (PoC[center])
                    CoG = self.getCoG(arr)

                    new_PoC[CoG] = []
                    new_centers.append (CoG)
                else:
                    new_PoC[center] = []
                    new_centers.append (center)

            '''print
            print "Old Centers", ' '.join (str (x) for x in centers)
            print "New Centers", ' '.join (str (x) for x in new_centers)'''

            result = []
            if (sorted(centers) == sorted (new_centers)):
                result = [list(x) for x in new_centers]
                break
            else:
                PoC = new_PoC
                centers = new_centers


        (aaa, dist) = self.squaredErrorDistorttion(data_points, centers)

        print "REst", dist
        print "Result"
        for r in result:
                print ' '.join ( str ("%0.3f" % x) for x in r)

        return ""

