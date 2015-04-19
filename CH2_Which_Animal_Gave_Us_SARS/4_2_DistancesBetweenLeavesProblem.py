__author__ = 'ilap'

from bioLibrary import *
'''
Definitions:
    @@@@ 4.2
    Distance matrix (NxN) D properties:
        = symmetric: for all i and j Dij = Dji
        = non negative: for all i and j, Dij >=0
        = triangle inequality: for all i, j, and k, Dij+Djk >= Dik

    * Parent (it's a node/vertex): Given a leaf j, only one node connect to j by an edge, denoted Parent (j).
    * Limb (it's an edge): An edge connecto a leaf to its parent is called limb (foag, vegtag)
    * Leaf: it does not have outbound connections.
    * Rooted tree: One node is a special node called root (node).
    * Unrooted tree: Trees without a designated root.

    * length of path dij(T): The sum of edges between tow leaf
    * fits: A weighted unrooted tree T *fits* a distance matrix D if d (i,j, T) = Dij for every pair of leaves i and j.

    Additive and non-additive D (distance matrix):
    * Additive: if there exists a tree that fits this matrix
    * Non-Additive: if there is no tree that fits this D matrix
    * Additive Term: sum of lengths of all edges along the path between leaves i and j in the additive tree adds to Dij.
        Every 3x3 matrix D is additive.

    * Check the fittnes of a tree to a D distance matrix: f the number of equations in a linear system of equations is
        smaller than or equal to the number of variables in the system, then a solution usually exists

    * Tree (D): If a matrix is additive, then there exists a unique simple tree fitting this matrix.
        Construct a Tree from additive matrix D
    * Minimum element of a matrix means, minimum off-diagonal element, i.e., a Vvalue Dij such that i != j.
'''

#### Main
N=32
text = '''0->38:7
1->52:8
2->43:6
3->37:6
4->40:12
5->53:12
6->47:14
7->35:12
8->58:7
9->39:8
10->55:7
11->32:6
12->48:15
13->49:7
14->33:7
15->59:14
16->56:15
17->58:6
18->44:15
19->45:15
20->36:8
21->42:13
22->51:5
23->57:5
24->60:14
25->46:12
26->34:10
27->50:15
28->32:13
29->54:9
30->61:8
31->41:15
32->28:13
32->33:15
32->11:6
33->32:15
33->34:6
33->14:7
34->26:10
34->33:6
34->35:10
35->7:12
35->34:10
35->36:13
36->37:13
36->20:8
36->35:13
37->38:10
37->3:6
37->36:13
38->37:10
38->39:14
38->0:7
39->38:14
39->40:10
39->9:8
40->39:10
40->41:7
40->4:12
41->40:7
41->42:12
41->31:15
42->21:13
42->41:12
42->43:14
43->2:6
43->42:14
43->44:13
44->45:14
44->18:15
44->43:13
45->46:9
45->19:15
45->44:14
46->45:9
46->25:12
46->47:7
47->46:7
47->48:15
47->6:14
48->12:15
48->49:7
48->47:15
49->13:7
49->50:12
49->48:7
50->49:12
50->51:6
50->27:15
51->50:6
51->22:5
51->52:7
52->53:15
52->1:8
52->51:7
53->54:15
53->52:15
53->5:12
54->53:15
54->29:9
54->55:6
55->54:6
55->56:6
55->10:7
56->57:12
56->55:6
56->16:15
57->23:5
57->56:12
57->60:13
58->8:7
58->59:5
58->17:6
59->15:14
59->61:8
59->58:5
60->57:13
60->24:14
60->61:5
61->30:8
61->60:5
61->59:8'''

graph = Graph (text)

"""
for v in graph:
    for w in v.getConnections():
        print ("(%s , %s:%s, Degree %d)" % (v.getId(), w.getId(), v.getWeight (w), v.getDegree ()) )
"""
dm = graph.distance_matrix ()
for row in dm:
    print " ".join(str(v) for v in row)

#distanceBetweenLeaves(graph)