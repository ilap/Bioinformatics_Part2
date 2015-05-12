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

#####0 1 2 3 4 5
a = [1,2,3,4,5,6]
b=34

print a[0:3], a[3:], a[0:3]+ [b]+ a[3:]
c=0
print not c

m = np.reshape(['0.0', '0.0','0.0', '0.0', '5.0', '5.0', '0.0', '5.0', '1.0', '1.0', '2.0', '2.0', '3.0', '3.0', '1.0', '2.0'], (4,4))

print m[1][3]
print '%.2f' % 1.334332456
print 2.00001 ** 2

dp =np.array([ 10., 3.])

arr = np.array([[ 1., 3.],[ 1.,  6.],[ 3.,  4.]])

print "IS IT IN?", dp in arr

a = [[]]*3
#a=[[1],[2],[3]]
print a
a[0] += [0]
a[0] += [0]
print a

arr = np.array(((2,2,3,4),(2,-2,5,6)))
print tuple(map(tuple, arr))

import decimal
a = decimal.Decimal("6.0535102")
print(round(a,3))


