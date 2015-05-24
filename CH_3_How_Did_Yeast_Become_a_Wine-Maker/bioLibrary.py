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

class Vertex:
    def __init__(self,key, age=0, use_age=False):
        self.use_age = use_age
        self.age = age
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self,nbr,weight=0):

        if self.use_age:
            weight = self.age - nbr.age
            #print "self ID, age, weight", self.age, weight
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def hasConnections(self):
        return len (self.connectedTo.keys ()) !=0

    def getId(self):
        return self.id

    def getAge(self):
        return self.age

    def getWeight(self,nbr):
        return self.connectedTo[nbr]

    def setWeight(self,nbr, cost):
        if self.connectedTo.has_key(nbr):
            self.connectedTo[nbr] = cost

    def getDegree (self):
        return len (self.connectedTo.keys())

    def isLeaf (self):
        return self.getDegree () == 1

    def getParent (self):
        return

    def removeConnection (self, nbr):
        if nbr in self.connectedTo:
            del(self.connectedTo[nbr])
        else:
            print "ERROR, removeConnection Not itm", nbr


class Graph:
    """
    The string list is a from->to:weight formatted string e.g. 0->1:3
    """
    def __init__(self, string_list='', directed=False):
        self.is_unrooted = False
        self.vertList = {}
        self.numVertices = 0
        self.directed = directed
        if string_list:
            adj_list = string_list.split ('\n')

            for line in adj_list:
                values = line.split("->")
                f = values[0]
                to_and_weight = values[1].split(":")
                t = to_and_weight[0]
                w = to_and_weight[1]

                self.addEdge (int (f),int (t),int (w))


    def addVertex(self,key, age=0,use_age=False):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key, age, use_age)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    '''
        input: 0,1,5
    '''
    def addEdge(self,f,t,cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)

        self.vertList[f].addNeighbor(self.vertList[t], cost)
        if self.directed:
            #print "DIRECTED", f,t, cost
            self.vertList[t].addNeighbor(self.vertList[f], cost)

    '''
        input: 0,1,6
    '''
    def insertVertex (self, f, t, n, cost=0, as_root=False):
        if f not in self.vertList or t not in self.vertList:
            print "ERROR"
            return
        weight = self.vertList[f].getWeight (self.vertList[t])

        if as_root:
            self.addEdge (n,f, cost)
        else:
            self.addEdge (f, n, cost)
        self.addEdge(n, t, weight - cost)
        self.vertList[f].removeConnection (self.vertList[t])
        if self.directed:
            self.vertList[t].removeConnection (self.vertList[f])

    def addDistantEdge (self, i, j, v, d):

        f = self.getVertex (i)
        t = self.getVertex(j)
        path = self.find_path (f, t)
        if not path:
            print "WARNING", f, t, d, v, "weight", f.getWeight (t)
            return -1

        res = 0
        for idx in range (0, len (path)-1):
            fv = path[idx]
            tv = path[idx+1]
            pres = res
            res += fv.getWeight (tv)

            idf = fv.getId ()
            idt = tv.getId ()
            #print ("(%s->%s:%s), result %d, prev res %d" % (i,j, d, res, pres))
            if res == d:
                return (idf,idt)
            elif res > d:
                #print "INSERT", idf, idt,v, d, "RES", d-pres
                self.insertVertex(idf, idt, v, d-pres)
                return (idf, v)

        #print "ERROR", i, v
        return (i, v)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

    def __len__(self):
        return self.numVertices

    def find_path(self, start_vertex, end_vertex, path=[]):
        """ find a path from start_vertex to end_vertex in graph """
        graph = self.vertList #__graph_dict
        #print "GRAPH, start vertex", start_vertex.getId ()

        path = path + [start_vertex]
        if start_vertex == end_vertex:
            #print "Same", path
            return path

        if start_vertex.getId () not in graph:
            return None


        for vertex in start_vertex.getConnections ():
            if vertex not in path:
                extended_path = self.find_path(vertex, end_vertex, path)
                if extended_path:
                    return extended_path
        return None

    def hasDistance (self, i, j, distance):
        res = 0
        #print "IJ", i, j
        path = self.find_path (i, j)

        if not path:
            return False
        print "PATH",
        for idx in range (0, len (path)-1):
            res += path[idx].getWeight (path[idx+1])
            print path[idx].getId (), path[idx+1].getId (), path[idx].getWeight (path[idx+1]), "NEW  ",
            if res == distance:
                print
                return True
        print
        return False

    def d (self, i, j):
        res = 0
        path = self.find_path (i, j)

        for idx in range (0, len (path)-1):
            res += path[idx].getWeight (path[idx+1])
        #print "Weight", i, j, res
        return res


    def distance_matrix (self):
        leaves = []
        for leaf in self:
            if leaf. isLeaf ():
                leaves.append(leaf)

        leaves.sort()

        N = len (leaves)
        distance = np.array (np.zeros ((N,N), dtype=int))
        for i in range (N):
            for j in range (i+1, N):

                f = leaves[i]
                t = leaves[j]
                val = self.d (f,t)
                #print "i", i, "val", val
                distance[i][j] = val
                distance[j][i] = val

        return distance


    def printNodes (self):
        #IDX = 1
        for v in self:
            #print IDX, "Node: ", v.getId (), "Len: ", len (self), v
            for w in v.getConnections():
                print("%s->%s:%.3f" % (v.getId(), w.getId(),abs(v.getWeight (w))))
                #print("%s->%s:%s" % (v.getId(), w.getId(),abs(float("{0:.3f}".format(v.getWeight (w))))))
                #print("%s->%s:%s    age:%s" % (v.getId(), w.getId(),v.getWeight (w), v.getAge ()))
            #IDX += 1
        return

    def buildParsimonyTree (text, num_of_leaves):

        pass




'''
Matrix class
'''

class Matrix:

    def __init__(self, string_list="", n=0, m=0, dtype=np.int16):

        self.row_len = n
        self.dimension = m # same as col_len

        #print "TYPE", type (string_list)
        #exit ()
        if type (string_list) == np.ndarray:
            self.matrix = deepcopy(string_list)
        elif type (string_list) == list:
            self.matrix = np.array (deepcopy(string_list))
        else:

            #print "SELF ROW", self.row_len, self.dimension
            if self.row_len and not string_list:
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

            #print "ARRAY", self.row_len, self.dimension, arr
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
        m = len (v)
        for j in range (m):
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
    def getCoG (self, data_points, m=0):
        if m == 0:
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

    '''
    E steps

    '''
    def centersToSoftClustersNewtonian (self, centers):
        #print "CENTERS", centers
        data_points = self.getMatrix()

        #print "MA", data_points
        #exit ()
        n = len (data_points)
        k = len (centers)

        hidden_matrix = np.array(np.zeros((k, n), dtype=float))

        for j in range (n):
            for i in range (k):
                distance = 1/(self.eucledianDistance(data_points[j], centers[i])**2)
                sum_val = sum ([ (1/self.eucledianDistance(data_points[j],x)**2) for x in centers])

                hidden_matrix[i][j] = distance/sum_val

        return hidden_matrix

    '''
    E-Step
    '''
    def centersToSoftClustersPartitionFunction (self, centers, stiffnes=1):
        data_points = self.getMatrix()

        #print "MA", data_points
        #exit ()
        n = len (data_points)
        k = len (centers)

        euler = math.e
        hidden_matrix = np.array(np.zeros((k, n), dtype=float))

        for j in range (n):
            for i in range (k):
                d = self.eucledianDistance(data_points[j], centers[i])
                distance = math.e ** -(stiffnes * d)
                sum_val = sum ([ (math.e ** -(stiffnes * self.eucledianDistance(data_points[j], x))) for x in centers])

                hidden_matrix[i][j] = distance/sum_val

        return hidden_matrix

    ''' M-Step
    '''
    def softClusterToCenters (self, hidden_matrix):

        data = self.getMatrix()
        n = len (data)
        k = len (hidden_matrix)

        result = [0.0]*k
        for i in range (k):
            dot_p = np.dot (hidden_matrix[i], data)
            result[i] = list (dot_p/sum (hidden_matrix[i]))
        #print type (data)
        #print type (hidden_matrix)

        return tuple (result)

    def kmeansSoftClustering (self, k, stiffnes, steps=10):

        centers = self.getMatrix()[0:k]

        step = 0
        while step < steps:
            #print "Step", step
            hidden_matrix = self.centersToSoftClustersPartitionFunction (centers, stiffnes=stiffnes)
            centers = self.softClusterToCenters( hidden_matrix)
            step += 1

        for center in centers:
            print ' '.join ([ ('%.3f' % x)for x in center])
        return centers

    def pearsonCorrelationCoefficient (self, vx, vy):

        m = len (vx)

        means_vx = sum (vx)/m
        means_vy = sum (vy)/m

        dividend = [float]*m
        divisor_x = [float]*m
        divisor_y = [float]*m

        ax = list (vx)
        ay = list (vy)
        for i in range (m):
            mx = ax[i]-means_vx
            my = ay[i]-means_vy
            dividend[i] = mx*my
            divisor_x[i] = mx**2
            divisor_y[i] = my**2

        return sum(dividend)/math.sqrt((sum(divisor_x)*sum (divisor_y)))

    def removeOffDiagonal (self, j):
        self.dimension -= 1
        self.matrix = np.delete (self.matrix, j, axis=0)
        self.matrix = np.delete (self.matrix, j, axis=1)

    def addOffDiagonal (self, j, values=0):
        self.dimension += 1
        self.matrix = np.insert (self.matrix, j, values=values, axis=0)
        self.matrix = np.insert (self.matrix, j, values=values, axis=1)

    def closestDistance (self, D, clusters):
            # distance...
            m = D.getMatrix ()
            N = len (m)
            min = sys.maxint

            for i in range (N):
                for j in range (i+1,N):
                    #self.__distance_between_clusters__(D,C[i],C[j])
                    Dij = D.getMatrix ()[i][j]
                    if min > Dij:
                        #print "CLI", i, j, clusters[i]
                        #print "CLJ", i, j, clusters[j]

                        min = Dij
                        vi = i
                        vj = j
            #print "Create node and reduce cluster", min, vi, vj
            return (min, vi, vj)

    def averageDistance (self, D):

            m = D.getMatrix ()
            N = len (m)

            sum_Di = [float (0.0)]*N
            sum_Dj = [float (0.0)]*N
            print "MIN", m.min ()
            print "MEANS", m.mean ()

            for i in range (N):
                sv = 0
                for j in range (i+1,N):
                    #self.__distance_between_clusters__(D,C[i],C[j])
                    Dij = D.getMatrix ()[i][j]
                    sv += Dij

                n_i = N-i

                print "I",i,n_i-1, n_i, "AVG", sv/n_i
                sum_Di[i] = sv/n_i
                sum_Dj[n_i-1] = sum_Di[i]

            print "DI", sum_Di
            print "DJ", sum_Dj

            #exit ()


            #return (avg, c1, c2)
    '''
    '''
    def hierarchicalClustering (self,is_distance_matrix=False, k=1, print_all=True):

        #
        """
            Initializaiton phase
        """
        N=len (self.getMatrix())
        G = Graph (directed=True)

        if is_distance_matrix:
            D = deepcopy(self)
        else:
            dm = self.makeDistanceMatrixUsingPearson()
            D = ClusterMatrix (dm, m=N,n=N)
        #print "MATRIXXXXXXX", dm
        #print D.getMatrix ()

        clusters = [(0,0)]*N
        for i in range (N):
            clusters[i] = (i,1)
            G.addVertex(i, use_age=True)

        new_clusters = [None]*N
        #print G.numVertices
        #print "C", clusters
        #print "MATRIX"
        #print D.getMatrix ()

        # Clusters
        # C = C1 + C2 Merge 1 and 3 to a new cluster
        #
        # del (D[1]), del D([2]), D.append (C)
        # [{0:0}, {1:1}, {2:2}, {3:3}] -step1-> [{0:0}, {2:2}, {4:1,3}]
        #
        # del (D[0]), del D([1]), D.append (C)
        # [{0:0}, {2:2}, {4:1,3}] -step1-> [{5:0,2}, {4:1,3}]
        # and so on...
        idx = N
        clen = len (clusters)
        while clen > k:
            #(age, c1, c2) =  self.averageDistance(D)
            (age, c1, c2) =  self.closestDistance(D, clusters)

            #print "D", D.getMatrix()



            #exit ()
            ## Merge first
            D.addOffDiagonal (clen, 0)
            (n1, v1) = clusters[c1]
            (n2, v2) = clusters[c2]
            nr_clus = v1 + v2

            tn1 = str (n1+1)
            tn2 = str (n2+1)

            if n1 >= N or n2 >= N:
                if n1 >= N:
                    tn1 = new_clusters[n1]
                if n2 >= N:
                    tn2 = new_clusters[n2]


            #print "##########################################################"
            #print idx, "C1, C2", c1, c2, "N1,N2", n1, n2, tn1, tn2
            #print clusters

            new_clusters.append(str (tn1) + " " + str (tn2))
            #print new_clusters

            clusters.append((idx,nr_clus))
            del (clusters[c2])
            del (clusters[c1])


            D.matrix[clen] = (D.matrix[c1]*v1+D.matrix[c2]*v2)/(v1+v2)
            D.matrix[:,clen] = (D.matrix[c1]*v1+D.matrix[c2]*v2)/(v1+v2)
            D.matrix[clen][clen] = 0

            # Remove the bigger index one first
            D.removeOffDiagonal (c2)
            D.removeOffDiagonal (c1)

            # Create
            Vc1 = G.getVertex(n1)
            Vc2 = G.getVertex(n2)
            G.addVertex (idx, age=age/2, use_age=True)

            Vc = G.getVertex(idx)

            G.addEdge(idx, n1, age/2)
            G.addEdge(idx, n2, age/2)
            #
            # G.printNodes()
            #print "MATRIX"
            #print D.getMatrix ()
            clen = len (clusters)
            idx +=1

        #print "clusters", clusters
        clusters_data = []
        for (idx, clust_len) in clusters:
            clusters_data.append (new_clusters[idx])

        return (G, clusters_data)

    def makeDistanceMatrixUsingDotProduct (self):
        pass

    def makeDistanceMatrixUsingPearson (self):

        arr = self.getMatrix()
        n = len (arr)

        D = np.array (np.zeros ((n,n), dtype=float))
        #print "n&D", n, D
        for i in range (n):
            #print "NEW:" ,
            for j in range (i,n):
                if i != j:
                    v = self.eucledianDistance (np.array(list (arr[i])),np.array (list (arr[j])))
                    v = '%.1f' % self.eucledianDistance (np.array(list (arr[i])),np.array (list (arr[j])))
                    D [i][j] = v
                    D [j][i] = v
        return D


