# -*- coding: utf-8 -*-
"""Created on Thu Apr 14 20:24:06 2016

@author: zhaoshiwen
@email: zhaoshiwen520@gmail.com

This is a demonstration of using MELD to analyze categorical data. In
this demonstration, we simulated 20 categorical variables each with
four different categories. We simulated 1000 observations from the
generalized Dirichlet latent variable model (see
https://arxiv.org/abs/1603.05324 for the details of the generation
process). 

"""

import numpy as np
import matplotlib.pyplot as plt
import MELD as MELD
reload(MELD)

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------- Generate simulation data -------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

p = 20 # the number of variables
d = 4 # the number of categories for each variable
k = 3 # the number of components

# --------------- simulate PHI and save --------------------------
alpha_lambda = np.array([0.5]*d)
Phi = np.zeros(p,dtype=object)
for j in range(p):
    Phi_j = np.zeros((k,d))
    for h in range(k):           
        Phi_j[h,:] = np.random.dirichlet(alpha_lambda)
    Phi[j] = Phi_j
PHI = np.zeros((p*d,k))
for j in range(p):    
    PHI[j*d:(j+1)*d,:] = np.transpose(Phi[j])    
np.savetxt('data/CTp20PHI.txt',PHI)


# --------------- simulate 1000 observations --------------------
n = 1000
alpha = np.array([0.1]*k)
Y = np.zeros((p,n))
M = np.zeros((p,n))
for i in range(n):
    x_i = np.random.dirichlet(alpha)
    for j in range(p):
        m_ji = np.random.multinomial(1,x_i)
        m_ji = np.nonzero(m_ji)
        M[j,i] = m_ji[0][0]
        Yji = np.random.multinomial(1,Phi[j][m_ji[0][0],:])
        Y[j,i] = np.nonzero(Yji)[0][0]
np.savetxt('data/CTp20Yn' + str(n) + '.txt',Y)
np.savetxt('data/CTp20Mn' + str(n) + '.txt',M)


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------- Perform fast moment estimation -------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------


# --------------- load data from the txt file --------------------
n = 1000
Y = np.loadtxt('data/CTp20Yn' + str(n) + '.txt')
(p,n) = Y.shape

# --------------- define the type of the variables ---------------
# 0: categorical
# 1: non-categorical with non-restrictive mean
# 2: non-categorical with positive mean
Yt = np.array([0]*p) 

# --------------- define other variables for estimation ----------
k = 3 # the number of components
S = 100 # maximum number of iterations


# ----------------------------------------------------------------
# --------- Estimation using second moment matrices --------------
# ----------------------------------------------------------------

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


# ----------------------------------------------------------------
# --------- Estimation using second and third moments ------------
# ----------------------------------------------------------------

# create an object of MELD class
myMELD = MELD.MELD(Y,Yt,k)

# initialize component paramters Phi
myMELD.initializePhi()

# calculate third moment tensors
myMELD.calM3()
myMELD.calM3_bar()

# initialize weight matrices to identity
myMELD.initializeWeight_M3()

# start to perform first stage estimation
ResultM3S1 = myMELD.estimatePhiGrad_M2M3(S, prt=True)

# recalculate weight matrix
myMELD.updateWeight_M2()
myMELD.updateWeight_M3()

# start to perform second stage estimation
ResultM3S2 = myMELD.estimatePhiGrad_M2M3(S, prt=True, step = 0.1)

# look at the convergence of objective function
plt.plot(ResultM3S1['Q3'])
plt.show()


# ------------------------------
# ------------------------------
# --- analyze the results ------
# ------------------------------
# ------------------------------

# plot convergence of objective function

