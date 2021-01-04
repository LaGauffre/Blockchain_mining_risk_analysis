# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:23:52 2021

@author: pierr
Function to compute ruin probabilities and expected revenue via MC simulations 
over an exponential time horizon when implementing the original selfish mining 
strategy of Eyal and Sirer
"""

# Function to compute ruin probabilities 
# and the revenue function for a miner who is 
# mining selfishly 
import numpy as np
import scipy as sc
import pandas as pd
import math as ma
from scipy.special import binom
from scipy.stats import poisson
from scipy.optimize import fsolve
import scipy.optimize
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from scipy.stats import norm
from scipy.stats import t as stu
from numpy.random import Generator, PCG64, SeedSequence

# The functions herafter tends to share the following
# parameters:
# - u is the initial reserve
# - t is the time horizon
# - b is the reward for finding a block
# - c is the operational cost
# - p is the share of computing power
# - q is the connectivity
# - λ is the block arrival intensity



# Test
# u, b, c, p, q, λ = 2, 5, 0.4, 0.1, 0.5, 2
# psi_u(u, b, c, p, λ), psi_selfish_u(u, b, c, p, q, λ)



# Function that simulate trajectories of the reward process
def U_N_t_ES(n, p, q, Z_0, rg):
    # n is the number of blocks discovered by the network (the value of N_T)
    # Sequence of block discovery
    # Z_0 is the initial state of the Markove chain
    # rg sets the seed of the random number generator
    ξ = rg.binomial(1, p, size = n)
    # Sequence of miner's choice
    ζ = rg.binomial(1, q, size = n)
    # Sequence of buffer values 
    # Z = Z_N_t_ES(n, ξ, Z_0)
    Z_ES, Z_AG = [0], [0]
    U_ES, U_AG = [0], [0]
    for k in range(n):
        ################################
        #Initialization of the buffer
        # if the buffer is empty
        if Z_ES[-1] == 0:
            U_ES.append(0)
            Z_ES.append(Z_ES[-1] + ξ[k])
        # If the buffer is empty and a fork is ongoing
        elif Z_ES[-1] == 0.5:
            Z_ES.append(0)
            if ξ[k] == 1:
                U_ES.append(2)
            elif ξ[k] == 0 and ζ[k] == 1:
                U_ES.append(1)
            elif ξ[k] == 0 and ζ[k] == 0:
                U_ES.append(0)
        # If the buffer contains 1 block
        elif Z_ES[-1] == 1:
            U_ES.append(0)
            #If Sam finds a block
            if ξ[k] == 1:
                Z_ES.append(Z_ES[-1] + ξ[k])
            else:
                Z_ES.append(0.5)
        # If the buffer contains 2 blocks
        elif Z_ES[-1] == 2:
            #If Sam finds a block
            if ξ[k] == 1:
                U_ES.append(0)
                Z_ES.append(Z_ES[-1] + ξ[k])
            # If not then he releases all the block and get rewarded
            else:
                U_ES.append(
                    sum(
                        np.ediff1d(
                            np.array(Z_ES[(np.where(
                                np.array(Z_ES) == 0)[0][-1]):len(Z_ES)]))==1))
                Z_ES.append(0)
        # If the buffer contains more than 2 blocks
        elif Z_ES[-1] > 2:
            U_ES.append(0)
            #If Sam finds a block
            if ξ[k] == 1:
                Z_ES.append(Z_ES[-1] + ξ[k])
            # If not then he releases all the block and get rewarded
            else:
                Z_ES.append(Z_ES[-1] + (ξ[k]-1))
        ############################
        if Z_AG[-1] == 0:
            U_AG.append(0)
            Z_AG.append(Z_AG[-1] + ξ[k])
        # If the buffer is empty and a fork is ongoing
        elif Z_AG[-1] == 0.5:
            Z_AG.append(0)
            if ξ[k] == 1:
                U_AG.append(2)
            elif ξ[k] == 0 and ζ[k] == 1:
                U_AG.append(1)
            elif ξ[k] == 0 and ζ[k] == 0:
                U_AG.append(0)
        # If the buffer contains 1 block
        elif Z_AG[-1] == 1:
            #If Sam finds a block
            if ξ[k] == 1:
                Z_AG.append(0)
                U_AG.append(2)
            else:
                Z_AG.append(0.5)
                U_AG.append(0)
        
    U_ES.append(0)
    U_AG.append(0)
    
    return({'U_ES':np.array(U_ES),'U_AG':np.array(U_AG)})



# Test
n, p, q, Z_0, rg = 10, 0.3, 0.5, 0, Generator(PCG64(14))
print(U_N_t_ES(n, p, q, Z_0, rg))


# Function that produce simulated data for exponential time horizon
def sim_data_self_T_ES(t, λ, p, q, Z_0, K, rg):
    T_exp = rg.exponential(t, K)
    N_t = np.concatenate([rg.poisson(λ * T_exp[k], 1) for k in range(K)])
    T = [np.append(np.append(0,np.sort(np.random.uniform(0, T_exp[k], N_t[k]))),T_exp[k]) for k in range(K)]
    U = [U_N_t_ES(N_t[k], p, q, Z_0, rg) for k in range(K)]
    return({'T_exp':T_exp, 'N_t':N_t, 'T':T, 'U':U})


# Function to compute the revenue function over an exponential time horiszon via MC simulations
def V_T_self_MC_ES(u, t, λ, p, q, b, c, sim_data):
         
    T_exp = sim_data['T_exp']
    K = len(T_exp)
    # Number of jumps up to time t
    N_t = sim_data['N_t']
    # Jump times when R_t > 0
    T = sim_data['T'] 
    # Reward associated to each jump
    U = sim_data['U']

    # Surplus at the end of the time horizon
    R_t_AG = np.array([u - c * T_exp[k] + b * sum(U[k]['U_AG']) for k in range(K)])
    R_t_ES = np.array([u - c * T_exp[k] + b * sum(U[k]['U_ES']) for k in range(K)])
    # Surplus at check times
    R_s_AG = [np.array([u - c * T[k][i] + b * sum(U[k]['U_AG'][0:i]) for i in range(N_t[k] + 2)]) for k in range(K)]
    R_s_ES = [np.array([u - c * T[k][i] + b * sum(U[k]['U_ES'][0:i]) for i in range(N_t[k] + 2)]) for k in range(K)]
    no_ruin_AG = np.array([np.all(R_s_AG[k] > 0) for k in range(K)])
    no_ruin_ES = np.array([np.all(R_s_ES[k] > 0) for k in range(K)])
    ψ_AG = 1 - np.mean(no_ruin_AG)
    ψ_ES = 1 - np.mean(no_ruin_ES)
    return({'AG_selfish': np.array([ψ_AG, np.sqrt(ψ_AG*(1-ψ_AG)/K)*norm.ppf(0.975), np.mean(R_t_AG * no_ruin_AG.astype('uint8')),\
                     np.std(R_t_AG * no_ruin_AG.astype('uint8'))/np.sqrt(K) * stu.ppf(0.975,K-1)]),
            'ES_selfish': np.array([ψ_ES, np.sqrt(ψ_ES*(1-ψ_ES)/K)*norm.ppf(0.975), np.mean(R_t_ES * no_ruin_ES.astype('uint8')),\
                     np.std(R_t_ES * no_ruin_ES.astype('uint8'))/np.sqrt(K) * stu.ppf(0.975,K-1)])})

# Test
# Parameter operational cost
# pkW, network_yearly_tW, pBTC, nBTC = 0.04, 77.78, 9938, 12.5
# u, t, λ, p, q,Z_0, b = 25000, 12, 6, 0.02, 0.1,0, pBTC * nBTC
# rg = Generator(PCG64(123))
# c, sim_data = p * pkW * network_yearly_tW*10**9 / 365 / 24, sim_data_self_T_ES(t, λ, p, q, Z_0, 10000, rg)
# V_T_self_MC_ES(u, t, λ, p, q, b, c, sim_data)
