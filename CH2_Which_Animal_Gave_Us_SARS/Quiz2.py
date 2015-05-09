__author__ = 'ilap'


from bioLibrary import *

# Q4
print "Question 4...."
print "#################################################################################"
N=4

text= '''0 20  9 11
20  0 17 11
9 17  0  8
11 11  8  0'''

m = Matrix (text, N,dtype=int)

print m.isAdditive()

matrix =  m.getMatrix()

print len (matrix), matrix
print m.getLimbLength(1)

# Q5
print "Question 5...."
print "#################################################################################"
N=4
text= '''0 14 17 17
14  0  7 13
17  7  0 16
17 13 16  0'''

m = Matrix (text, N,dtype=int)

print m.isAdditive()

matrix =  m.getMatrix()

print len (matrix), matrix
print m.getNeighborJoiningMatrix()

# Q5
