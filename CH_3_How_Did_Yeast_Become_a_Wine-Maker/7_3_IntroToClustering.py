__author__ = 'ilap'

from bioLibrary import *


### Clustering Intro
k=3
m=2
points='''0.0 0.0
5.0 5.0
0.0 5.0
1.0 1.0
2.0 2.0
3.0 3.0
1.0 2.0
'''
''

matrix = ClusterMatrix (points, m=m, dtype=float)
result = matrix.farthestFirstTraversal(k)
print "RESULT"
for arr in result:
    print ' '.join ([str ('%.1f' % val) for val in arr])


