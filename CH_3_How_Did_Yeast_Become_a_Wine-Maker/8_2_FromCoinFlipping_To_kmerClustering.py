__author__ = 'ilap'

from bioLibrary import *

datas=[
[1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
[1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
[1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
[0, 1, 1, 1, 0, 1, 1, 1, 0, 1]]

'''
 data corresponds the percentage of the heads means
 data_i=count (1 in datas_i)/len (datas_i)
'''
data=(0.4, 0.9, 0.8, 0.3, 0.7)
# Hidden vector for parameters, if we know parameters...
'''
HiddenVector corresponds to determining whether coin A or coin B was more likely to have generated
the n observed flips in each of the five coin flipping sequences.
'''
hidden_vector=(1, 0, 0, 1, 0)

'''
For example, suppose we know that Parameters = (thetaA , thetaB) = (0.6, 0.82).
If coin A was used to generate the fifth sequence of flips (with seven heads and three tails),
then the probability that coin A generated this outcome is
'''

'''
Parameters = (theta_A, theta_B)
    thetaA = sum (hidden_vector * data_i)
    thetaA = sum ((1-hidden_vector) * data_i)
* means dot product
'''
parameters=(0.35, 0.8)

######
''' #Example1 (Data, HiddenVector, ?) -> Parameters
Data: Known (0.4, 0.9, 0.8, 0.3, 0.7)
Hidden_Vector: Not Known
Parameters: Known (0.6, 0.82)

Solution: hv_i = theta_A^(data_i*10)*(1-theta_A)^((1-data_i)*10)
'''
def getHiddenVectorFromParameters (data, parameters):

    p = list (parameters)
    k = len (p)

    d = list (data)
    n = len (data)

    #print p


    result = [0]*n
    thetas = [float (0)]*k
    for i in range (n):

        r1=int (d[i]*10)
        r2=int ((1-d[i])*10)

        for j in range(k):
            theta_i = p[j]
            thetas[j] = theta_i**r1*(1-theta_i)**r2

        #print "K", k, "thetas", thetas
        max_idx = thetas.index(max(thetas))
        result[i] = abs (j - max_idx)
        #print "MAX_IDX", abs (j - max_idx), result[i]
        #else:
        #    hv = 0
    return tuple (result)

print "getHiddenVectorFromParameters", getHiddenVectorFromParameters(data, parameters)


''' #Example1 (Data, Parameters, ?) -> HiddenVector
Data: Known (0.4, 0.9, 0.8, 0.3, 0.7)
Hidden_Vector: Known (1, 0, 0, 1, 0)
Parameters: Not Known

Solution:
Parameters = (theta_A, theta_B)
    thetaA = sum (hidden_vector * data_i)
    thetaB = sum ((1-hidden_vector) * data_i)
* means dot product
'''
def getParametersFromHiddenVector (data, hidden_vector):
    d = np.array (list (data))
    hv_a = np.array(list (hidden_vector))
    hv_b = 1-np.copy(hv_a)

    theta_A = float (np.dot (hv_a, d)/sum (hv_a))
    theta_B = float (np.dot (hv_b, d))/sum (hv_b)
    #print "A", theta_B
    #print "B", np.dot (hv_b, d)/3
    #print "C", sum (hv_b * d)/3
    return (theta_A, theta_B)

print "getParametersFromHiddenVector", getParametersFromHiddenVector(data, hidden_vector)

''' #Example3 (Data, ?, ?) -> ??????? How?
Random guess. We will therefore start from an arbitrary choice of Parameters = (theta_A,theta_B)
and immediately reconstruct the most likely HiddenVector:
(Data, ?, Parameters) -> HiddenVector
As soon as we know HiddenVector, we will question the wisdom of our initial choice of
Parameters and re-estimate Parameters':
(Data, HiddenVector, ?) -> Parameters
'''

def getAllFromData (data):

    parameters = (0.6, 0.82)
    old_parameters = (0.0, 0.0)

    while parameters != old_parameters:
        hv = getHiddenVectorFromParameters(data, parameters)
        #print "HV", hv
        old_parameters = getParametersFromHiddenVector(data, hv)
        #print "OP", old_parameters

        tmp_parameters = parameters
        parameters = old_parameters
        old_parameters = tmp_parameters


    return parameters


print "getAllFromData", getAllFromData(data)
