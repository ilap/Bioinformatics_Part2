__author__ = 'ilap'


from bioLibrary import *

N=4
text= '''0 3 4 3
3 0 4 5
4 4 0 2
3 5 2 0'''

m = Matrix (text, N,dtype=float)

print m.isAdditive()

matrix =  m.getMatrix()

print len (matrix), matrix
print m.matrix

