__author__ = 'ilap'

from bioLibrary import *
#### MAIN

text = "panamabananas$"
(arr, str) = text2SuffixArray(text)

print "ARR", arr
print "STR", str
print "NEW LINE"
print text2BurrowsWheeler(text)

print text2BurrowsWheeler2(text)

