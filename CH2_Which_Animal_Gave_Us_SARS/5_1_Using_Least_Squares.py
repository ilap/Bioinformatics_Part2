__author__ = 'ilap'

from bioLibrary import *


'''
Definitions:
    @@@@ 4.4
    * Bald limb: the row and column subtracted by Limb Length
    * Bald D: Obtained from D whereeach off-diagonal element in the jth row and column of D is substracted by Lib Length.
        where jbecome bald limb or limb of 0 length
    * Trimmed D: D distance matrix ignoring bald limb (removing jth row/column) nxn D --> (n-1)(n-1)Dtrimmed

'''
N=4

text ='''0 3 4 3
3 0 4 5
4 4 0 2
3 5 2 0'''

m = Matrix (text)

m.discrepancy()




