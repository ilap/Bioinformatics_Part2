__author__ = 'ilap'

from bioLibrary import *

##########
m=2
k=2
centers = np.array ([[-2.5,0],[2.5, 0]])
data_points ='''-3 0
-2 0
0 0
2 0
3 0'''


matrix = ClusterMatrix (data_points,m=m, dtype=np.float16)

print matrix.centersToSoftClustersNewtonian(centers)
print matrix.centersToSoftClustersPartitionFunction (centers, stiffnes=0.5)
print matrix.centersToSoftClustersPartitionFunction (centers, stiffnes=1)


m2=2
k2=3
centers2 = np.array ([[3, 4.5],[9,5],[6, 1.5]])
data_points2 ='''1 3
1 6
3 4
5 2
5 6
7 1
8 7
10 3'''
matrix = ClusterMatrix (data_points,m=m, dtype=np.float16)

print matrix.centersToSoftClustersNewtonian(centers)
print matrix.centersToSoftClustersPartitionFunction (centers, stiffnes=0.5)
hm =  matrix.centersToSoftClustersPartitionFunction (centers, stiffnes=1)
print hm
print matrix.softClusterToCenters (hm)