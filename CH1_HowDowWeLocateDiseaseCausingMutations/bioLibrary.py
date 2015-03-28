__author__ = 'ilap'

## Imports
import threading

import sys
sys.setrecursionlimit(1000000000)

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
    #TODO sort does not need
    ##  patterns = sorted (patterns)

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


'''
nonBranching...
'''
def deletePath (graph, from_node, to_node=-1):
    lock = threading.Lock()

    lock.acquire() # will block if lock is already held
    if from_node in graph:
        nodes = graph[from_node]
        if len (nodes) > 1:
            if to_node == -1:
                idx = 0
            else:
                #print "NOEDS", nodes
                idx = nodes.index (to_node)
            del graph[from_node][idx]
        else:
            del graph[from_node]
    lock.release()
    return graph

def outPath (graph, node):

    result = -1
    if graph.has_key(node):
        result =  len (graph[node])
    return result


def inPath (graph, node):
    i = 0
    for graph_keys in graph.keys ():
        values = graph[graph_keys]
        i += values.count (node)
    return i

def oneInOneOut (graph, node):
    i1 = inPath(graph, node)
    o1 = outPath(graph, node)
    #print "INOUT-----", node,  i1, o1, (i1 == o1) and (i1 == 1)
    return ( (i1 == o1) and (i1 == 1))

def maximalNonBranchingPaths(graph, start_node=-1, end_node=-1):
    paths = []

    if start_node != -1 and end_node != -1:
        deletePath(graph, start_node, end_node)

    import copy
    new_graph = copy.deepcopy (graph)

    for node in graph.keys ():
        if not oneInOneOut(graph, node):
            non_branch_path = ""
            if outPath(graph, node) > 0:
                for n in graph[node]:

                    non_branch_path = str(node) + " -> "+ str(n)
                    deletePath(new_graph, node, n)
                    w = n

                    while oneInOneOut(graph, w):

                        u = graph[w][0]
                        non_branch_path += " -> "+ str (u)
                        #print "NBP2", non_branch_path
                        deletePath(new_graph, w, u)
                        w = u
                    paths.append(non_branch_path)
        #print graph
        #print "FINAL", new_graph


    if not len (new_graph):
        return paths


    nodes = new_graph.keys ()

    indegreezero = []
    for node in nodes:
        if inPath(new_graph, node):
            indegreezero.append (node)
        else:
            print "######### WARNINGGGGGG ###########", node

    #print "SSSSSSSSSSSSSSS", indegreezero
    #print "NEWGRAPH", new_graph

    for start_node in indegreezero:
        new_cycle = ""
        snode = start_node


        while True:
            new_cycle += str (snode) + " -> "
            if not new_graph.has_key(snode):
                break

            onode = snode

            snode = new_graph[snode][0]

            #print "NW", snode, onode, new_cycle
            deletePath(new_graph, onode)
            if start_node == snode:
                new_cycle += str (snode)
                paths.append (new_cycle)
                #print "BREAKED", new_cycle
                new_cycle = ""
                break


        if not new_graph:
            break

    return paths