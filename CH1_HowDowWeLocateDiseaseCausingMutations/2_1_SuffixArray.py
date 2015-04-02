__author__ = 'ilap'

from bioLibrary import *
'''
2.1 Burrows-Wheeler transform is a lexicographically ordered cyclic rotations
    and the last character of that matrix
'''
@timing
def text2BurrowsWheeler (text):

    import bisect
    burrows_wheeler_array = []
    str_len = len (text)

    for i,c in enumerate (text):
        cycle = text[i:]+text[0:i]
        #bisect.insort(burrows_wheeler_array, cycle )
        burrows_wheeler_array.append(cycle)


    burrows_wheeler_array.sort ()
    #print '\n'.join (burrows_wheeler_array)
    result = ''.join (x[str_len-1] for x in burrows_wheeler_array)

    return result

@timing
def text2BurrowsWheeler2 (text):

    import bisect
    burrows_wheeler_array = []
    str_len = len (text)

    for i,c in enumerate (text):
        cycle = text[i:]+text[0:i]
        bisect.insort(burrows_wheeler_array, cycle )



    #print '\n'.join (burrows_wheeler_array)
    result = ''.join (x[str_len-1] for x in burrows_wheeler_array)

    return result
'''
2.1 Suffix Array
'''
@timing
def text2SuffixArray (text):
    suffix_array = []
    for i,c in enumerate (text):
        suffix_array.append ((text[i:], i))

    suffix_array.sort ()

    return suffix_array
#### MAIN

text = "panamabananas$"

print ', '.join ([str (x) for s,x in text2SuffixArray(text)])

print text2BurrowsWheeler(text)

print text2BurrowsWheeler2(text)

