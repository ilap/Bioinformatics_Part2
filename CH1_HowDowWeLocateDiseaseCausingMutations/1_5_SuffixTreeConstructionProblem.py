__author__ = 'ilap'

from bioLibrary import *





### MAIN
#### WORKS withe Extra DATA set and /w normal...

STR = "panamabananas$"
STR = "ATATCGTTTTATCGTT$"
tr =  modifiedSuffixTrieConstruction(STR)
sft =  rebuildSuffixTrie(tr)
gr = compressedSuffixTrie2Graph(sft)
print suffixTreeMatching(sft, STR)
