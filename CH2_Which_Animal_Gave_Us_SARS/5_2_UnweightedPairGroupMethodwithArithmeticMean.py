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

N1=3
text1 ='''0 3 3.5
3 0 4.5
3.5 4.5 0'''

N=9
text ='''0	295	300	524	1077	1080	978	941	940
295	0	314	487	1071	1088	1010	963	966
300	314	0	472	1085	1088	1025	965	956
524	487	472	0	1101	1099	1021	962	965
1076	1070	1085	1101	0	818	1053	1057	1054
1082	1088	1088	1098	818	0	1070	1085	1080
976	1011	1025	1021	1053	1070	0	963	961
941	963	965	962	1057	1085	963	0	16
940	966	956	965	1054	1080	961	16	0'''
m = Matrix (text, dtype=float)

m.upgma (N).printNodes ()

