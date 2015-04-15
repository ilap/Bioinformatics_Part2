__author__ = 'ilap'

import  numpy as np
from bioLibrary import *

arr1 = [1,2,3]


print arr1

@timing
def a ():
    res = ''
    for j in range (0, 65):
        for i,c in enumerate (text):
            res = c
    return res

@timing
def b ():

    #clen = len (text)
    res = ''
    for j in range (0, 65):
        for i in range (0, clen):
            res = text[i]

    return res

text = ""
clen = len (text)
a()
b()

c = [ str (x) for x in []]
c.sort ()
print ' '.join (c)

A='''ACACCTTTTGCACACGATACATGCACGATCTGCGTGCGCCAAAACACCGTATGGTTGCATGTGTGC
GGCGATCATAGACCATGTAGGGAGCCACAGTAGCATCGGGATCGTCGGATCGGCGATGC
2'''

B = A.split('\n')
print int (B[2])