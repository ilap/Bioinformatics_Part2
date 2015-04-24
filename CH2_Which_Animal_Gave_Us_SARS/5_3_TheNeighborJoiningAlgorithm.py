__author__ = 'ilap'

from bioLibrary import *


'''
Definitions:

'''
N=4
text ='''0 13 21 22
13 0 12 13
21 12 0 13
22 13 13 0'''

m = Matrix (text, dtype=float)

m.getNeighborJoiningMatrix ()

