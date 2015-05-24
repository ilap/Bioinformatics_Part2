__author__ = 'ilap'


from bioLibrary import *

''' Plot sample
plt.ioff()
plt.plot(np.random.rand(10))
plt.show()
'''

'''YLR258W = np.matrix((1.07, 1.35, 1.37, 3.70, 4.00, 10.00, 5.88))
YPL012W = np.matrix((1.06, 0.83, 0.90, 0.44, 0.33, 0.13, 0.12))
YPR055W = np.matrix((1.11, 1.11, 1.12, 1.06, 1.05, 1.06, 1.05))

print np.log2 (YLR258W)
print np.log2 (YPL012W)
print np.log2 (YPR055W)

exit ()

plots = np.array ([YLR258W, YPL012W, YPR055W])
print plots

f=-6
t=6
s=2

plt.ioff ()
plt.xlim (f,t)
plt.ylim (-5, 15,1)
plt.yticks(range (-5,15,1))

print range (f,t+s,2)
plt.plot (range (f,t+s,2), plots[0], color="green")
plt.plot (range (f,t+s,2), plots[1], color="red")
plt.plot (range (f,t+s,2), plots[2], color="blue")
plt.show()'''

k=2
m=2
points='''2 8
2 5
6 9
7 5
5 2
'''

centers= np.array ([[3, 5],[5, 4]])
print centers


matrix = ClusterMatrix (points, m=2, dtype=float)

print "MAXDIST", matrix.maxDataPointsDistanceToCenters(matrix.getMatrix(), centers)

p = np.array ([[2,6],[4,9],[5,7],[6,5],[8,3]])
c = np.array ([[4,5],[7,4]])


print "DIST", matrix.squaredErrorDistorttion(p, c)

p = np.array ([[1, 3, -1], [9, 8, 14], [6, 2, 10], [4, 3, 1]])
print "Center of G", matrix.getCoG(p, m=3)
