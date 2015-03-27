__author__ = 'ilap'

from bioLibrary import *

##### MAIN
# TESTED tr = trieConstruction ("ananas", "and", "anternna", "banana", "bandana", "nab", "nana", "pan")
# TESTED print "TR", tr
# TESTED print prefixMatching("panamabananas", tr)

matches = treeMatching("AATCGGGTTCAATCGGGGT", "ATCG", "GGGT")
print " ".join ([ str (x) for x in matches])


