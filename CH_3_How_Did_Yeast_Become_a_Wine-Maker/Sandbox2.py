__author__ = 'ilap'

from bioLibrary import *

data=np.array(
[[0.00, 0.74, 0.85, 0.54, 0.83, 0.92, 0.89],
 [0.74, 0.00, 1.59, 1.35, 1.20, 1.48, 1.55],
 [0.85, 1.59, 0.00, 0.63, 1.13, 0.69, 0.73],
 [0.54, 1.35, 0.63, 0.00, 0.66, 0.43, 0.88],
 [0.83, 1.20, 1.13, 0.66, 0.00, 0.72, 0.55],
 [0.92, 1.48, 0.69, 0.43, 0.72, 0.00, 0.80],
 [0.89, 1.55, 0.73, 0.88, 0.55, 0.80, 0.00]])

tot = 0.

m=7
print "XR", data.shape[0]

tot = .0
for i in range (m):
    c1 = data[i][i+1:]
    c2 = data[i][:i]
    print "C12", c1, c2
    tot += sum (c1)
    print len (c1), len (c2)
print tot**2/m