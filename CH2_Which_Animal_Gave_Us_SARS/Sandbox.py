__author__ = 'ilap'

import numpy as np

a = np.array([[ 0,  6,  8,  7,  7,  8,  7],
 [ 6,  0, 10,  9,  9, 10,  9],
 [ 8, 10,  0,  3,  9, 10,  9],
 [ 7,  9,  3,  0,  8,  9,  8],
 [ 7,  9,  9,  8,  0,  5,  4],
 [ 8, 10, 10,  9,  5,  0,  5],
 [ 7,  9,  9,  8,  4,  5,  0]])
j = 6
print "BEF", j
print a
a = np.delete(a, j, 0)
a = np.delete (a, j, 1)


print "AFT", j
print a

