__author__ = 'ilap'

from bioLibrary import *

'''
2.5 Pattern match with Burrows-Wheeler Transwform
'''
# @timing
def patternMatchInBWT (text, pattern):

    text_len = len (text)

    first_col = []
    last_col_table = {}

    for i,ch in enumerate (text):
        if not last_col_table.has_key(ch):
            last_col_table[ch] = [i]
        else:
            last_col_table[ch].append (i)

        first_col.append( (ch,i))

    first_col.sort()
    #print first_col
    first_col_text = ''.join(sorted (text))
    #print first_col_text


    top = 0
    bottom = len (text)-1

    occurance = 0

    for i in  range (len (pattern)-1, -1, -1):
        curr_char = pattern [i]
        #print curr_char

        if not last_col_table.has_key(curr_char):
            occurance = 0
            break

        import bisect
        fc_tidx = bisect.bisect_left(last_col_table[curr_char],top)
        fc_bidx = bisect.bisect_right(last_col_table[curr_char],bottom)

        occurance = fc_bidx - fc_tidx


        if occurance != 0:
            top = bisect.bisect_left(first_col_text, curr_char)+fc_tidx
            bottom = top + occurance - 1
            #print "OCC", occurance, curr_char, top, bottom
        else:
            occurance = 0
            break

    return occurance
### MaAIN
# DONE string = "am$al"
# DONE print reconstructTextFromBWT(string)

text = "smnpbnnaaaaa$a"
print reconstructTextFromBWT(text)
patterns = ["ana"]
result = []
for pattern in patterns:
    result.append(str (patternMatchInBWT (text, pattern)))

print ' '.join (result)



