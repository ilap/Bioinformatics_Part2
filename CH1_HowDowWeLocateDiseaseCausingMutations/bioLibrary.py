__author__ = 'ilap'

'''
########################################################################################
####### 1.5 Suffix Trees
########################################################################################
'''



'''
########################################################################################
####### 1.3 Herding Patterns To Tries
########################################################################################
'''
'''
TRIECONSTRUCTION(Patterns)
        Trie <- a graph consisting of a single node root
        for each string Pattern in Patterns
            currentNode <- root
            for i <- 1 to |Pattern|
                if there is an outgoing edge from currentNode with label currentSymbol
                    currentNode <- ending node of this edge
                else
                    add a new node newNode to Trie
                    add a new edge from currentNode to newNode with label currentSymbol
                    currentNode <- newNode
        return Trie
'''
'''
Trie Construction Problem: Construct a trie on a set of patterns.
     Input: A collection of strings Patterns.
     Output: Trie(Patterns).
     Note: Checked/Tested...
'''
def trieConstruction (*patterns):

    # Sort them first
    patterns = sorted (patterns)

    trie = [{}] # Root, with no edges and nodes
    for word in patterns:
        current_node =  0

        for idx, current_symbol in enumerate (word):

            nodes = trie[current_node]
            if nodes.has_key(current_symbol):
                current_node = nodes[current_symbol]
            else:
                new_node = len (trie)
                trie.append({})
                trie[current_node][current_symbol] = new_node
                print str(current_node) + "->" + str (nodes[current_symbol]) +":" + current_symbol
                current_node = new_node

    return trie

'''
PREFIXTRIEMATCHING(Text, Trie)
        symbol <- first letter of Text
        v <- root of Trie
        while forever
            if v is a leaf in Trie
                return the pattern spelled by the path from the root to v
            else if there is an edge (v, w) in Trie labeled by symbol
                symbol <- next letter of Text
                v <- w
            else
                output "no matches found"
                return
'''
def prefixMatching (text, trie):

    str_idx = 0
    current_symbol = text[str_idx]
    current_node = 0;
    result = ""

    while True:

        #next_nodes = {v: k for k, v in trie[current_node].items()}

        if trie[current_node] == {}:
            return result
        else:

            if trie[current_node].has_key (current_symbol):

                result += current_symbol

                current_node = trie[current_node][current_symbol]

                str_idx += 1
                try:
                    current_symbol = text[str_idx]
                except:
                    #print "Exception"
                    True
            else:
                #print "No Matches Found!"
                return
'''
TRIEMATCHING(Text, Trie)
        while Text is nonempty
            PREFIXTRIEMATCHING(Text, Trie)
            remove first symbol from Text
'''
def treeMatching (text, *patterns):



    trie = trieConstruction(*patterns)
    result = []
    str_idx = 0
    while str_idx < len (text):

        match = prefixMatching(text[str_idx:], trie)
        if match != None:
            result.append (str_idx)
        str_idx += 1

    return result
