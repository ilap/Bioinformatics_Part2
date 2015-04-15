__author__ = 'ilap'

from bioLibrary import *

### MaAIN
# Q8
print "#### Q8"
string = "AT$AAACTTCG"
print reconstructTextFromBWT(string)

# Q4
print "#### Q4"
string = "adnaadnanadadnan$"
pattern = "ad"
bwt =  text2BurrowsWheeler(string)
print patternMatchInBWT (bwt, pattern)

# Q5
print "### Q5"
string = "GCCAGCTCTTTCAGTATCATGGAGCCCATGG$"
print "Leaves == len of string", len (string)

# Q7
print "### Q7"
string = "GATTGCTTTT$"
print text2BurrowsWheeler (string)

string = "GCCAGCTCTTTCAGTATCATGGAGCCCATGG$"
print "@@@@ QUICK"
#string = "panamabananas$"
bwt_str = text2BurrowsWheelerQuick(string)
#print "BWT_STRING", bwt_str

CURR_CHAR = bwt_str[0]
RUN_LEN = 10
CURR_LEN = 0
FOUND = 0

#bwt_str = string
print bwt_str[:20]
for ch in bwt_str:
    CURR_LEN += 1
    if CURR_CHAR != ch:
        print "DIFFER", CURR_CHAR, CURR_LEN, ch
        if CURR_LEN > RUN_LEN:
            print "HEUREKA", CURR_CHAR, CURR_LEN
            #CURR_LEN = 0
            FOUND += 1
        CURR_CHAR = ch
        CURR_LEN = 0

print "FOUND", FOUND
