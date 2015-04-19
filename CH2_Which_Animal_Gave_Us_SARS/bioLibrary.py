__author__ = 'ilap'

import numpy as np
import sys
from copy import *
'''
Vertex (Node)
A vertex (also called a "node") is a fundamental part of a graph. It can have a name, which we will call the "key".
A vertex may also have additional information. We will call this additional information the "payload".

Edge (link)
An edge (also called an "arc") is another fundamental part of a graph. An edge connects two vertices to show that
there is a relationship between them. Edges may be one-way or two-way.
If the edges in a graph are all one-way, we say that the graph is a directed graph, or a digraph.
The class prerequisites graph shown above is clearly a digraph since you must take some classes before others.

Weight
Edges may be weighted to show that there is a cost to go from one vertex to another. For example in a graph of
roads that connect one city to another, the weight on the edge might represent the distance between the two cities.

The graph abstract data type (ADT) is defined as follows:

    * Graph() creates a new, empty graph.
    * addVertex(vert) adds an instance of Vertex to the graph.
    * addEdge(fromVert, toVert) Adds a new, directed edge to the graph that connects two vertices.
    * addEdge(fromVert, toVert, weight) Adds a new, weighted, directed edge to the graph that connects two vertices.
    * getVertex(vertKey) finds the vertex in the graph named vertKey.
    * getVertices() returns the list of all vertices in the graph.
    * in returns True for a statement of the form vertex in graph, if the given vertex is in the graph, False otherwise.

Reference http://interactivepython.org/runestone/static/pythonds/Graphs/graphintro.html
'''
class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]

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


    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
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
            self.vertList[t].addNeighbor(self.vertList[f], cost)

    '''
        input: 0,1,6
    '''
    def insertVertex (self, f, t, n, cost=0):
        if f not in self.vertList or t not in self.vertList:
            print "ERROR"
            return
        weight = self.vertList[f].getWeight (self.vertList[t])
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
                print("%s->%s:%s" % (v.getId(), w.getId(),v.getWeight (w)))
            #IDX += 1
        return



class Matrix:

    def __init__(self, string_list, dimension, dtype=np.uint16):

        self.matrix = []
        self.original_dimension = dimension
        self.length = dimension
        self.matrix = np.fromiter (string_list.split (), dtype)
        self.matrix = np.reshape(self.matrix, (dimension,dimension))

    def getMatrix(self):
        return self.matrix

    def removeOffDiagonal (self, j):
        self.length -= 1
        self.matrix = np.delete (self.matrix, j, axis=0)
        self.matrix = np.delete (self.matrix, j, axis=1)

    '''
    Limb Length Problem: Compute the length of a limb in a tree defined by an additive distance matrix.
        Input:  An additive distance matrix D and an integer j.
        Output: LimbLength(j), the length of the limb connecting leaf j to its parent.
    '''
    def getLimbLength (self, j):

        min = sys.maxint
        steps = 0
        for i in range (0, self.length):
            if i != j:
                for k in range (i+1, self.length):
                    if j != k:
                        ###steps += 1
                        val = (self.matrix[i][j] + self.matrix[j][k] - self.matrix[i][k])/2
                        if val < min:
                            min = val
        ####print "Steps", steps
        return  min

    def AdditivePhylogeny (self, N):

        G = Graph (directed=True)
        D = deepcopy (self)

        T = D.__additivePhylogeny__(G, N, len (self.matrix)-1)
        return T

    def __additivePhylogeny__ (self, G, N, node_idx=0):
        node_idx += 1
        N -= 1
        if N == 1:
            G.addEdge (0, 1, self.matrix[0][1])
            return G

        limb_length = self.getLimbLength (N)

        self.getMatrix()[N] -= limb_length
        self.getMatrix()[:,N] -= limb_length
        self.getMatrix()[N,N] = 0

        ith = -1
        kth = -1

        for i in range (0, N+1):
            for k in range (i+1, N+1):
                #print "IJ", i, k
                Dik = self.matrix[i,k]
                DiN = self.matrix[i][N]
                DNk = self.matrix[N][k]

                if  Dik == (DiN+ DNk):
                    #print node_idx, limb_length, "i,k,n:", i, k, N, " Dik DiN DNk", Dik, DiN, DNk
                    x = self.matrix[i,N]
                    ith = i
                    kth = k
                    #print "HEUREKA", i, k, N, x
                    break
            else:
                continue
            break

        # Make Trimmed D, remove [N,N]

        self.removeOffDiagonal(N)

        T = self.__additivePhylogeny__(G, N, node_idx)
        (f, v) = T.addDistantEdge (ith, kth, node_idx, x)

        #print "BEF: ####", node_idx, T.getVertices ()
        #T.printNodes ()
        #print ("ADD (%d->%d:%d)" % (v, N, limb_length))
        T.addEdge (v, N, limb_length)
        #print "AFT: ####", node_idx,  T.getVertices ()
        #T.printNodes ()
        if node_idx != 11:
            if node_idx != 10:
                if node_idx != 9:
                    pass
                    #exit ()


        return T
