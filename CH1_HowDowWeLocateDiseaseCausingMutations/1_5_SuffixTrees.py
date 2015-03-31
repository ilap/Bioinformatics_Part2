__author__ = 'ilap'

from bioLibrary import *

##### MAIN
'''
graph = trie2Graph(tr)
# maximalNonBranchingPaths(graph)
# WORKED WITH BIG DATA SET, AND PROBLEM SET, but it's changed, CHECK IT AGAIN
#### print '\n'.join ( ' -> '.join (map (str, v)) for v in maximalNonBranchingPaths(graph))
'''

#STR = "panamabananas$"
STR = "ATCTACCAGCAGTGAACATGGGAGGACCAGTAAGGAAGGCTTACCCTCGATGTGTTACAGACTCGTTCGTAGGGTGTATAACGCCGCCGCTGG$"
tr =  modifiedSuffixTrieConstruction(STR)
sft =  rebuildSuffixTrie(tr)
sufixTreeConstructionProblem (sft, STR)


