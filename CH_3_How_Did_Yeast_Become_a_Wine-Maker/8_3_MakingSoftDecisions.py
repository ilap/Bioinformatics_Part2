__author__ = 'ilap'

from bioLibrary import *
'''
Introduction (see prev chapter for more details_
#################################################################
Given n data points in m-dimensional space Data = (Data1, ... , Datan), we represent their
assignment to k clusters as an n-dimensional vector

HiddenVector = (HiddenVector1, ... , HiddenVectorn),
where each HiddenVector_i can take integer values from 1 to k.

We will then represent the k centers as k points in m-dimensional space,
Parameters = (theta_1, . . . , theta_k).
In k-means clustering, similarly to the coin flipping analogy, we are given Data,
but HiddenVector and Parameters are unknown.

The Lloyd algorithm starts from randomly chosen Parameters, and we can now rewrite its
two main steps as follows:

Centers to Clusters: (Data, ?, Parameters) -> HiddenVector
Clusters to Centers: (Data, HiddenVector, ?) -> Parameters
'''
'''
Data = (Data1, ... , Datan)
'''
data=(0.4, 0.9, 0.8, 0.3, 0.7)
# Hidden vector for parameters, if we know parameters...
'''
HiddenVector = (HiddenVector1, ..., HiddenVector)
'''
hidden_vector=(1, 0, 0, 1, 0)

'''
Parameters = (theta_1, ..., theta_k)
'''
parameters=(0.6, 0.82)

'''
# E-Step: Instead of Hidden vector the result is the HiddenMatrix
'''
def getHiddenMatrixFromParameters (data, parameters):

    p = list (parameters)
    k = len (p)

    d = list (data)
    n = len (data)

    #print p

    hidden_matrix = np.array(np.zeros((k, n), dtype=float))

    thetas = [float (0)]*k

    for i in range (n):

        r1=d[i]*10
        r2=(1-d[i])*10
        for j in range(k):
            theta_i = p[j]
            thetas[j] = theta_i**r1*(1-theta_i)**r2

        sum_val = sum (thetas)

        for j in range(k):
             hidden_matrix[j][i] = thetas[j]/sum_val

    return hidden_matrix

hidden_matrix = getHiddenMatrixFromParameters(data, parameters)
print "getHiddenMatrixFromParameters", hidden_matrix

'''
M-Step: Get Parameters form HiddenMatrix
'''
def getParametersFromHiddenMatrix (data, hidden_matrix):

    d = np.array (list (data))
    n = len (d)
    k = len (hidden_matrix)

    parameters = [float (0)]*k
    for i in range (k):
        hidden_vector = hidden_matrix[i]
        hv_i = np.array(list (hidden_vector))
        #print "HV_I", hv_i
        #exit ()
        #print "SUMi", i, sum (hidden_vector)

        theta_i = float (np.dot (hv_i, d)/sum (hidden_vector))
        parameters[i] = theta_i
        #print "%0.4f" % theta_i

    return tuple (parameters)

print "getParametersFromHiddenVector", getParametersFromHiddenMatrix (data, hidden_matrix)

def getAllFromData (data):

    parameters = (0.6, 0.82)
    old_parameters = (0.0, 0.0)

    while parameters != old_parameters:
        hv = getHiddenMatrixFromParameters(data, parameters)
        #print "HV", hv
        old_parameters = getParametersFromHiddenMatrix(data, hv)
        #print "OP", old_parameters

        tmp_parameters = parameters
        parameters = old_parameters
        old_parameters = tmp_parameters


    return parameters


print "getAllFromData", getAllFromData(data)



