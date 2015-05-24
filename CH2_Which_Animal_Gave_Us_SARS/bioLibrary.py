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



class Matrix:


    def __init__(self, string_list="", dimension=0, dtype=np.int16):

        if not string_list:
            if dimension == 0:
                print "ERROR", exit ()
            else:
                arr = [0]*dimension*dimension
        else:
            arr = string_list.split ()


        import math
        self.matrix = []

        if dimension == 0:
            dimension = int (math.sqrt (len (arr)))

        self.original_dimension = dimension
        self.length = dimension
        self.matrix = np.fromiter (arr, dtype)
        self.matrix = np.reshape(self.matrix, (dimension,dimension))

    def getMatrix(self):
        return self.matrix

    def removeOffDiagonal (self, j):
        self.length -= 1
        self.matrix = np.delete (self.matrix, j, axis=0)
        self.matrix = np.delete (self.matrix, j, axis=1)

    def addOffDiagonal (self, j, values=0):
        self.length += 1
        self.matrix = np.insert (self.matrix, j, values=values, axis=0)
        self.matrix = np.insert (self.matrix, j, values=values, axis=1)

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
                Dik = self.matrix[i,k]
                DiN = self.matrix[i][N]
                DNk = self.matrix[N][k]

                if  Dik == (DiN+ DNk):
                    x = self.matrix[i,N]
                    ith = i
                    kth = k
                    break
            else:
                continue
            break

        # Make Trimmed D, remove [N,N]
        self.removeOffDiagonal(N)
        T = self.__additivePhylogeny__(G, N, node_idx)
        (f, v) = T.addDistantEdge (ith, kth, node_idx, x)
        T.addEdge (v, N, limb_length)

        return T

    def isAdditive (self):

        return self.__four_point_theorem__()

    def isUnrootedBinary (self):
        return  self.is_unrooted
    '''
    Four Point Theorem:
    A distance matrix is additive if and only if the four point condition holds for every quartet (i, j, k, l) of
    indices of this matrix.
    1. Di,j + Dk,l
    2. Di,k + Dj,l
    3. Di,l + Dj,k

    Four Point Conditions:
    1. Two of the sums are equal and
    2. 3rd sum is less then or equal to the other two sums.

    1. (D1==D2 or D2==D3 or D1==D3) and
    2. (D3 <= (D1+D2))

    '''

    def __four_point_theorem__ (self):

        res = False

        D= self.matrix
        mlen = len (D)
        N=3 # starting from 0

        """
        from itertools import combinations_with_replacement, combinations, permutations, islice
        perm = range (0,mlen)
        all_perms = combinations_with_replacement (perm, 2)
        perms_arr = [x for x in all_perms]
        """


        for i in range (mlen-N):
            for j in range (i, mlen-N):
                Dij = D[i][j+1]
                Dik = D[i][j+2]
                Dil = D[i][j+3]
                Djk = D[i+1][j+2]
                Djl = D[i+1][j+3]
                Dkl = D[i+2][j+3]

                D1 = Dij + Dkl
                D2 = Dik + Djl
                D3 = Dil + Djk
                #print D1, D2, D3
                res = ((D1==D2 or D2==D3 or D1==D3) and (D3 <= (D1+D2)))

                if not res:
                    print "Failed at", i,j
                    break

        return res

    def __build_tree__ (self, G):

        for i in range (self.length):
            for j in range(i+1,self.length):
                G.addEdge (i,j,self.matrix[i][j])


    def discrepancy (self):
        T = Graph ()
        self.__build_tree__(T)
        TM = Matrix (dimension=self.length)
        FT = Graph ()

        for i in range (self.length):
            for j in range(i+1,self.length):
                vi = T.getVertex (i)
                vj = T.getVertex(j)
                dij = T.d (vi,vj)
                Dij = self.matrix[i][j]
                dval = (dij - Dij)^2
                print "DDD", dij, Dij, dval
                TM.getMatrix()[i][j] = (dij - Dij)^2
                TM.getMatrix()[j][i] = (dij - Dij)^2

                #FT.addEdge (i,j,self.matrix[i][j])
        #print "TM IS ", TM.getMatrix()
        #print TM.isAdditive()
        #T.printNodes()

    '''
    UPGMA (which stands for Unweighted Pair Group Method with Arithmetic Mean) is a simple clustering heuristic that
    introduces a hypothetical molecular clock for constructing an ultrametric
    evolutionary tree. You can learn more about clustering in a later chapter.

    Given an nxn matrix D, UPGMA (which is illustrated in the figure on the next step) first forms n trivial clusters,
    each containing a single leaf.
    The algorithm then looks for a pair of "closest" clusters.
    To clarify the notion of closest clusters, UPGMA defines the distance between clusters
    C1 and C2 as the average pairwise distance between elements of C1 and C2:

    Steps:
    1. build a trivial cluster, each containing a single leaf
    2. Then it looks for the closest clusters.
    3. The closes clusters uses upgma algoriths.
    Dc1c2 = Sum (i e c1 Sum j e c2 * Dij)/ (number of leaves in C1 * number of leaves in C2)
    '''
    def upgma (self, N):

        #
        """
            Initializaiton phase
        """
        G = Graph (directed=True)
        D = deepcopy (self)
        clusters = [(0,0)]*N
        for i in range (N):
            clusters[i] = (i,1)
            G.addVertex(i, use_age=True)
        print G.numVertices
        print "C", clusters
        print "MATRIX"
        print D.getMatrix ()

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
        while clen > 1:
            (age, c1, c2) =  self.closestDistance(D, clusters)
            print "AGE, C1, C2", age, c1, c2
            ## Merge first
            D.addOffDiagonal (clen, 0)
            (n1, v1) = clusters[c1]
            (n2, v2) = clusters[c2]
            nr_clus = v1 + v2
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
        return G

    '''
    Input: Distance Matrix
    Output: min value, index of the two closest array.
    '''
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

    '''
    Input: Distance Matrix
    Output: min value, index of the two closest array.
    '''
    def getNeighborJoiningMatrix (self):
        # distance...
        M = self.getMatrix ()
        Dstar = M.copy ()

        dlen = len (Dstar)

        total_distances = [0]*dlen
        for i in range (dlen):
            total_distances[i]= sum (M[i])

        min_i = sys.maxint
        min_j = sys.maxint
        val = sys.maxint
        for i in range (dlen):
            for j in range (i+1,dlen):

                Dval = (dlen-2)*M[i][j]-total_distances[i]-total_distances[j]
                print "D*ij", i,j, ": ", Dval
                Dstar[i][j] = Dval
                Dstar[j][i] = Dval
                if Dval < val:
                    #print "MIN", Dval, i,j
                    val = Dval
                    min_i = i
                    min_j = j


        return (Dstar, total_distances, min_i, min_j)

    '''
    Limb Length Problem: Compute the length of a limb in a tree defined by an additive distance matrix.
        Input:  An additive distance matrix D and an integer j.
        Output: LimbLength(j), the length of the limb connecting leaf j to its parent.
    '''
    def getLimbLengthFromNeighborJoining (self, neighbor_joining, total_distances, j):
        #NM stands for Neighbor Joining Matrix
        min = sys.maxint
        steps = 0
        njm = neighbor_joining
        N = len (njm)
        for i in range (0, N):
            if i != j:
                for k in range (i+1, N):
                    if j != k:
                        Deltaij = (total_distances[i]-total_distances[j])/(N-2)
                        val = 0.5*(njm[i][j]+Deltaij)
                        print "VAL", val, min
                        if val < min:
                            min = val
        ####print "Steps", steps
        return  min

    def getMinimumelements (self, njm):

        min = sys.maxint
        steps = 0
        #njm = self.getMatrix()

        N = len (njm)
        min_i = 0
        min_j = 0
        for i in range (0, N):
                for j in range (i+1, N):
                    if i != j:
                        val = njm[i][j]
                        print "NJM", i, j, njm[i][j]
                        if val < min:
                            min = val
                            min_i = i
                            min_j = j
        ####print "Steps", steps
        return  (min_i,min_j)


    def neighborJoining (self,N):
        G = Graph (directed=True)
        D = deepcopy (self)
        columns = range (N)
        T = D.__neighbor_joining__(G, N, columns, len (self.matrix)-1)
        return T

    def compressMatrix (self, min_i, min_j):

        M = self.getMatrix()
        N = len (M)

        for i in range (0, N):
            if i != min_i and i != min_j:
                val = (M[i][min_i]+M[i][min_j]-M[min_i][min_j])/2
                #print "Dim:", i,min_i, "VAL", val
                M[i][min_i] = val
                M[min_i][i] = val


        self.removeOffDiagonal(min_j)

        return  self.getMatrix()



    def __neighbor_joining__ (self, G, N, columns, node_idx=0):
        node_idx += 1
        N -= 1
        if N == 1:
            #print "COLUMNSF", node_idx, columns[0], columns[1], columns
            G.addEdge (columns[0], columns[1], self.matrix[0][1])
            return G

        (Dstar, total_distances, min_i, min_j) = self.getNeighborJoiningMatrix()
        delta = (total_distances[min_i]- total_distances[min_j])/(N-1)
        limb_i = (self.getMatrix()[min_i][min_j]+delta)/2
        limb_j = (self.getMatrix()[min_i][min_j]-delta)/2

        print "LIMB", N+1, N-1, "Delta", delta, limb_i, limb_j

        #Compress...
        compressed = self.compressMatrix(min_i, min_j)


        #print "COMPRESSED"
        #print compressed

        fn = node_idx
        ti = columns[min_i]
        tj = columns[min_j]

        G.addVertex (fn)
        G.addVertex (ti)
        G.addVertex (tj)

        #print "COLUMNSA", fn, ti, tj, columns
        columns[min_i] = node_idx
        del (columns[min_j])

        T = self.__neighbor_joining__ (G, N, columns, node_idx)

        #print "LEAFS", fn, ti, tj
        T.addEdge (fn, ti, limb_i)
        T.addEdge (fn, tj, limb_j)


        '''self.removeOffDiagonal(N)
        T = self.__additivePhylogeny__(G, N, node_idx)
        (f, v) = T.addDistantEdge (ith, kth, node_idx, x)
        T.addEdge (v, N, limb_length)'''

        return T

class ParsimonyVertex (Vertex):
    def __init__(self,key, age=0, use_age=False):
        Vertex.__init__(self,key, age=age, use_age=use_age)
        self.states = []
        self.scores = {}
        self.label = ""

    def setLabel (self, label):
        self.label = label

    def getLabel (self):
        return self.label

    def hasLabel (self):
        return self.label != ""

    def addState (self, state):
        self.states.append (state)

    def getState (self, pos):
        return self.states[pos]

    def getStates (self):
        return self.states

    def addScore (self, alphabet, score):
        self.scores[alphabet] = score

    def getScore (self, alphabet):
        return self.scores[alphabet]

    def getScores (self):
        return self.scores

    def setTag (self, tag):
        self.age = tag

    def getTag (self):
        return self.age

    def isRipe (self):
        """ We call internal node of T ripe if
            * its tag is 0 but
            * its children's tags are both 1.
        """
        res = False
        if self.getDegree():
            res = self.getTag() == 0
            #print "SELF", self.getId(), res
            for children in self.getConnections():
                res &= children.getTag () == 1
                #print "CHLD", res, children.getId (), children.getTag ()
        return res

    def isNumber(self):
        s = self.id
        try:
            float(s)
            return True
        except ValueError:
            try:
                from fractions import Fraction
                Fraction(s)
                return True
            except ValueError:
                return False

class ParsimonyGraph (Graph):
    def __init__(self,string_list='', num_of_leaves=0, un_rooted=False, directed=False, alphabet=['A','C','G','T']):
        Graph.__init__(self, directed=directed)
        self.leaves = {}
        self.root_node = None
        self.num_of_leaves = num_of_leaves
        self.alphabet = alphabet
        self.start_index = 1000

        if string_list:
            adj_list = string_list.split ('\n')

            for line in adj_list:
                #print "LINE", line
                if line != "":
                    values = line.split("->")
                    f = values[0]
                    t = values[1]
                    #print "ADD EDGE", f, t
                    #print "---"
                    lab = ""
                    si = self.start_index
                    if f.isalpha():
                        if self.hasLabel(f):
                            continue
                        lab = f
                        f = si
                    elif t.isalpha():
                        if self.hasLabel(t):
                            continue
                        lab = t
                        t = si

                    self.start_index +=1
                    self.addEdge (f, t)
                    if lab != "":
                        v = self.getVertex(si)
                        v.setLabel (lab)
                        self.leaves[si] = v


    def hasLabel (self, label):

        res = False
        for v in self.getVertices ():
            if self.getVertex(v).getLabel () == label:
                res = True
        return res

    def addVertex(self,key, age=0,use_age=False):
        print "ADDDING VERTEX", key ,
        self.numVertices = self.numVertices + 1
        newVertex = ParsimonyVertex(key, age, use_age)

        self.vertList[key] = newVertex
        return newVertex


    def smallParsimony (self):

        node = self.leaves.values()[0]

        label_len = len (node.getLabel ())
        SP = 0
        for i in range (label_len):
            (tsp, root_node) = self.__small_parsimony__(i)
            if self.root_node == None:
                self.root_node = root_node.getId ()
            print "##### MIN", i, tsp, root_node.getId()
            SP += tsp
            self.__backTrack__(i, tsp, parent_node=None, node=root_node)
            #exit ()
            #break
        print "SmallParsimony ###########################"
        print SP



    def __Dik__ (self, i,k):
        if i == k:
            return 0
        else:
            return 1

    def __small_parsimony__ (self, i):

        #print "## proces Small Parsimony"
        if self.directed:
            sub = 1
        else:
            sub = 0
        for node in self.getVertices():

            v = self.getVertex(node)
            # For leaves, the degree is 0 in non directed graph and 1 in directed ones
            if v.getDegree() - sub == 0:
                v.setTag (1)
                for k in self.alphabet:
                    character = v.getLabel()[i]
                    #print "CHAR", character, k
                    if character == k:
                        v.addScore (k, 0)
                    else:
                        v.addScore (k, sys.maxint)
                #print "LEAF SCORE", v.getLabel (), v.getScores()
            else:
                v.setTag (0)


        is_more_ripe = True
        root_node = None
        while is_more_ripe:

            is_more_ripe = False
            for node in self.getVertices():

                v = self.getVertex(node)
                if v.isRipe ():
                    min_res = sys.maxint
                    #print "V is RIPE", v.getId ()
                    is_more_ripe = True
                    v.setTag (1)
                    root_node = v


                    children = v.getConnections ()

                    daughter = children[0]
                    son = children [1]
                    #print "DAU SCORE", daughter.getScores()
                    #print "SON SCORE", son.getScores()

                    for k in self.alphabet:

                        min_d = []
                        min_s = []

                        for i in self.alphabet:
                            min_d.append (daughter.getScore (i) + self.__Dik__(i,k))
                            min_s.append (son.getScore (i) + self.__Dik__(i,k))
                        #print "MIN_D", min_d
                        #print "MIN_S", min_s
                        min_val = min (min_d) + min (min_s)

                        #print "MIN VAL", k, min_val, min_res

                        if min_val < min_res:
                            #print "Min has found", min_val
                            min_res = min_val
                        v.addScore (k, min_val)

                    print "SCORES", v.getId (), "MIN", min_val, v.getScores (), "RROT_NODE", root_node.getId ()

        return (min_res, root_node)

    def printNodes (self, print_chars=False, print_all=False, print_states=True):
        print "HAAAAHOOOOOO", print_chars
        if not print_chars:
            Graph.printNodes(self)
        else:
            for v in self:

                if print_all and not v.hasConnections():
                    print("STATES %s->(%s)" % (v.getId(), v.getStates ()))
                    print("%s->NOCONN (%s)" % (v.getId(), v.getScores ()))
                for w in v.getConnections():
                    print("STATES %s->(%s)" % (v.getId(), v.getStates ()))
                    print("%s->%s:%.3f (%s)" % (v.getId(), w.getId(),abs(v.getWeight (w)), v.getScores ()))
                    #print("%s->%s:%s" % (v.getId(), w.getId(),abs(float("{0:.3f}".format(v.getWeight (w))))))
                    #print("%s->%s:%s    age:%s" % (v.getId(), w.getId(),v.getWeight (w), v.getAge ()))
                #IDX += 1
        return

    def printParsimony (self, remove_root=False):

        if remove_root:
            self.removeRootVertex()

        for v in self:
            v_dna = ''.join([str (x) for x in v.getStates ()])
            for w in v.getConnections() :
                w_dna = ''.join([str (x) for x in w.getStates ()])
                hd = self.hammingDistance(v_dna, w_dna)
                print("%s->%s:%d" % (v_dna, w_dna, hd))
                print("%s->%s:%d" % (w_dna, v_dna, hd))

        return


    def __backTrack__ (self, i, tsp, parent_node=None, node=None):

        if not node:
            return None
        if not parent_node:
            s_root = node.getScores ()
            state = min (s_root, key=s_root.get)
            node.addState (state)
        else:


            s_j = parent_node.getScores ()
            min_val = min(s_j.itervalues())
            s_j_states = [k for k, v in s_j.iteritems() if v == min_val]

            s_i = node.getScores ()
            min_val = min(s_i.itervalues())
            s_i_states = [k for k, v in s_i.iteritems() if v == min_val]

            s_i = None
            for s_j in s_j_states:
                if s_j in s_i_states:
                    s_i = s_j
                    break
            if not s_i:
                s_i = s_i_states[0]

            node.addState (s_i)

            print "SJ STATES", s_j_states
            print "SI STATES", s_i_states
            print "SI", s_i
            pass


        left = None
        right = None
        parent = node
        children = node.getConnections ()

        if children:
            left = children[0]
            right = children [1]

        self.__backTrack__(i, tsp, parent, left)
        self.__backTrack__(i, tsp, parent, right)
        return

    def hammingDistance (self, p, q):
        mismatch = 0
        plen = len (p)
        qlen = len (q)
        if qlen != plen:
            return -1

        for i in range (0, plen):
           # print "PD: ", p[i], q[i]
            if p[i] != q[i]:
                mismatch += 1

        return mismatch

    def makeRootedFromUnroot (self):

        if not self.directed:
            print "Graph is not directed"
            return

        from random import randint

        conns = len(self.getVertices())


        pickup = randint(0,conns-1)
        print "CONNS" ,conns, "RND", pickup, self.getVertices()[pickup]
        f_idx = self.getVertices()[pickup]
        f = self.getVertex(f_idx)

        t = f.getConnections ()[0]
        print "CONNS,", t
        t_idx = t.getId ()

        r_idx = self.start_index
        self.start_index += 1



        self.insertVertex(f_idx,t_idx,r_idx, as_root=True)
        root_node = self.getVertex(r_idx)
        self.__make_root_directed__(parent_node=None, node=root_node)
        self.directed = False

        return

    def __make_root_directed__ (self, parent_node=None, node=None):

        if not node:
            return None
        if parent_node:
            # remove connection from children
            #print "BEF NODE", parent_node, node
            node.removeConnection(parent_node)
            #print "AFT NODE", parent_node, node


        left = None
        right = None
        parent = node
        children = node.getConnections ()

        #print "CHILDREN", children
        clen = len (children)

        if clen > 1:
            left = children[0]
            right = children [1]

        self.__make_root_directed__(parent, left)
        self.__make_root_directed__(parent, right)
        return

    '''
        input: 0,1,6
    '''
    def removeRootVertex (self):

        r = self.getVertex(self.root_node)
        children = r.getConnections ()

        f = children[0]
        t = children [1]

        r.removeConnection (f)
        r.removeConnection (t)

        self.addEdge(f.getId (), t.getId ())


