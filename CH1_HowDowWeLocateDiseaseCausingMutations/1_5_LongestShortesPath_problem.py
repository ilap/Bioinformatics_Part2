__author__ = 'ilap'

from bioLibrary import *

def checkChildrensColor (suffix_trie, text, node = 0):

    nodes = suffix_trie[node].keys ()
    first_color = ''
    color = ''

    for tmp_node in nodes:
        ((str_pos, str_len), (idx, color)) = suffix_trie[node][tmp_node]
        #print "COLOR IN CHECK", c_idx, color
        if first_color == '':
            first_color = color
        else:
            #print "CCC", first_color, color
            if color == 'g':
                color = checkChildrensColor(suffix_trie, text, tmp_node)
            if first_color != color:
                color = 'p'
                break


    return color


def treeColoring  (suffix_trie, text, node = 0, result=""):

    nodes = suffix_trie[node].keys ()
    hash_idx = text.index('#')
    #print "NODES to walk", nodes
    for tmp_node in nodes:
        ((str_pos, str_len), (idx, color)) = suffix_trie[node][tmp_node]
        new_result = text[str_pos:str_pos+str_len]
        #print "NODE", tmp_node, node, result

        nr_keys = suffix_trie[tmp_node].keys ()
        key_len = len (nr_keys)
        if  nr_keys != []:
            #print "means NOT LEAF", hash_idx, str_pos, str_len, new_result, idx, color
            #print "AAA", node, tmp_node, " AAA", key_len, result, " NEW R:", new_result
            #new_result = result + new_result
            #if (max_len < len (new_result)):
            #    max_len = len (new_result)
            new_result = result + new_result
            treeColoring(suffix_trie, text, tmp_node,new_result)
            color = checkChildrensColor (suffix_trie, text, tmp_node )

            #print "COLOR", color, new_result

        else:
            if idx <= hash_idx:
                color = 'b'
            else:
                color = 'r'
            suffix_trie[node][tmp_node] = ((str_pos, str_len), (idx, color))
            #print "Means LEAF", hash_idx, str_pos, str_len, new_result, idx, color

        if color == 'p':
            print len(new_result), new_result
    return

### MAIN


STR1 = "panama#"
STR2 = "bananas$"
STR1 = "CCAAGCTGCTAGAGG$"
STR2 = "CATGCTGGGCTGGCT$"

STR = STR1+STR2

tr1 =  modifiedSuffixTrieConstruction(STR1)
tr2 = modifiedSuffixTrieConstruction(STR2)
csft1 =  rebuildSuffixTrie(tr1)
csft2 =  rebuildSuffixTrie(tr2)

gr1 = compressedSuffixTrie2Graph(csft1)
print suffixTreeMatching(csft1, STR1)

gr2 = compressedSuffixTrie2Graph(csft2)
print suffixTreeMatching(csft1, STR2)
exit ()
#gr = compressedSuffixTrie2Graph(csft)

#treeColoring(csft, STR)
