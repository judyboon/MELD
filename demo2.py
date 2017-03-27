# -*- coding: utf-8 -*-
"""Created on Thu Apr 14 20:24:06 2016

@author: zhaoshiwen
@email: zhaoshiwen520@gmail.com

This is a demonstration of using MELD to analyze mix data tpyes. In
this demonstration, we consider a setting mimicking applications in
which DNA sequence variations influence a quantitative trait. We
generate a sequence of nucleotides {A, C, G, T} at 50 genetic loci
along with a continuous or integer-valued trait, resulting in p = 50+1
variables. Among the 50 nucleotide loci, eight of them are assumed to
be associated with the trait. We test whether MELD is able to detect
the eight loci. The details of this demonstration can be found in
paper https://arxiv.org/abs/1603.05324.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import MELD as MELD
reload(MELD)

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------- Generate simulation data -------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

p = 50 # the number of categorical variables
d = 4 # the number of categories for each categorical variable
k = 2 # the number of componentsn = 1000
n1 = 500 # first component has 500 observations
n2 = 500 # second component has the same number
n = n1 + n2 # total number of observations

# the eight loci that are assumed to be associated with the trait
J = np.array([2, 4, 12, 14, 32, 34, 42, 44])
J = J - 1
k=2

# --------------- simulate PHI and save --------------------------
alpha_lambda = np.array([0.5]*d)
Phi = np.zeros((p,k,d))
for j in range(p):
    if j in J:
        for h in range(k):
            Phi[j,h,:] = np.random.dirichlet(alpha_lambda)
    else:
        prob = np.random.dirichlet(np.array([1.0]*d))
        for h in range(k):
            Phi[j,h,:] = prob
PHI = np.zeros((p*d,k))
for j in range(p):
    for h in range(k):
        PHI[(j*d):((j+1)*d),:] = np.transpose(Phi[j,:,:])
np.savetxt('data/QTLp50PHI.txt',PHI)

# --------------- simulate 1000 observations --------------------
phi_u = np.array([5,10]) # the parameter for Poisson distribution
#phi_u = np.array([-3, 3]) # the parameter for Gaussian distribution
alpha = np.array([0.1]*k)
Y = np.zeros((p+1,n))
M = np.zeros((p+1,n))
Phi = np.zeros((p,k,d))
PHI = np.loadtxt('data/QTLp50PHI.txt')
for j in range(p):
    for h in range(k):
        Phi[j,:,:] = np.transpose(PHI[(j*d):((j+1)*d),:])
    
for i in range(n):
    x_i = np.random.dirichlet(alpha)
    for j in range(p):
        m_ji = np.random.multinomial(1,x_i)
        m_ji = np.nonzero(m_ji)
        M[j,i] = m_ji[0][0]
        Y_ji = np.random.multinomial(1,Phi[j][m_ji[0][0],:])
        Y[j,i] = np.nonzero(Y_ji)[0][0]
    m_ji = np.random.multinomial(1,x_i)
    m_ji = np.nonzero(m_ji)
    M[p,i] = m_ji[0][0]
    Y[p,i] = np.random.poisson(phi_u[m_ji[0][0]])
    #Y[p,i] = np.random.normal(phi_u[m_ji[0][0]])
np.savetxt('data/QTLp50Yn' + str(n) + '.txt',Y)
np.savetxt('data/QTLp50Mn' + str(n) + '.txt',M)


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------- Perform fast moment estimation -------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------


# --------------- load data from the txt file --------------------
Y = np.loadtxt('data/QTLp50Yn' + str(n) + '.txt')


# --------------- define the type of the variables ---------------
# 0: categorical
# 1: non-categorical with non-restrictive mean
# 2: non-categorical with positive mean
Yt = np.array([0]*(p+1))
Yt[p] = 2 # for Poisson trait
# Yt[p] = 1 # for Gaussian trait


# --------------- define other variables for estimation ----------
k = 2 # the number of components
S = 100 # maximum number of iterations


# --------- estimation using second moment matrices --------------

# create an object of MELD class
myMELD = MELD.MELD(Y,Yt,k)

# initialize component paramters Phi
myMELD.initializePhi()

# calculate second moment matrices
myMELD.calM2()
myMELD.calM2_bar()

# initialize weight matrices to identity
myMELD.initializeWeight_M2()

# start to perform first stage estimation
# set prt to True to print iterations
ResultM2S1 = myMELD.estimatePhiGrad_M2(S, prt=True)

# recalculate weight matrix
myMELD.updateWeight_M2()

# start to perform second stage estimation
ResultM2S2 = myMELD.estimatePhiGrad_M2(S, prt=True, step = 0.1)

# look at the convergence of objective function
plt.plot(ResultM2S1['Q2'])
plt.show()



# --------------- define the type of the variables ---------------
# 0: categorical
# 1: non-categorical with non-restrictive mean
# 2: non-categorical with positive mean
Yt = np.array([0]*p) 

# --------------- define other variables for estimation ----------
k = 3 # the number of components
S = 100 # maximum number of iterations


# --------- estimation using second moment matrices --------------

# create an object of MELD class
myMELD = MELD.MELD(Y,Yt,k)

# initialize component paramters Phi
myMELD.initializePhi()

# calculate second moment matrices
myMELD.calM2()
myMELD.calM2_bar()

# initialize weight matrices to identity
myMELD.initializeWeight_M2()

# start to perform first stage estimation
# set prt to True to print iterations
ResultM2S1 = myMELD.estimatePhiGrad_M2(S, prt=True)

# recalculate weight matrix
myMELD.updateWeight_M2()

# start to perform second stage estimation
ResultM2S2 = myMELD.estimatePhiGrad_M2(S, prt=True, step = 0.1)

# look at the convergence of objective function
plt.plot(ResultM2S1['Q2'])
plt.show()




#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------------- Analyze the results ------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

# ---------------- calculate average KL divergence ---------------

S = 100
k = 2
KL = np.zeros(p)
for j in range(p):
    counts = np.unique(Y[j,],return_counts=True)[1]
    marginal = counts*1.0/n
    for h in range(k):
        probh = ResultM2S1['PHI'][S-1][j][h,:]
    KL[j]= KL[j] + entropy(probh,qk=marginal)
KL = KL/k


# -------------------- plot KL divergence ------------------------
# the loci with largest average KL divergence are associated with the
# trait
plt.plot(KL)
plt.show()
