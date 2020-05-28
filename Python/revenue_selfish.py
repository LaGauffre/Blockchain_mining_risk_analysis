# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:35:57 2020

@author: pierr
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

# Function to compute the average profit when mining selfishly 
def η(p, q, b, λ, π_W, W): 
    # pkW price per kW
    # Electricity consumption of the network per year     
    c = p * π_W * W
    denom = 1+2*p-p**2
    p_U = [(1+p*(1-p)+p*(1-p)**2*(1-q))/denom, (p*q*(1-p)**2)/denom, (p**2+p**2*(1-p))/denom]
    return b*λ*(p_U[1]+2*p_U[2]) - c

# Test 
# p,q,b,λ,pkW,network_yearly_tW = 0.01, 0.5, 86796.25, 6, 0.04, 70
# η(p,q,b,λ,pkW,network_yearly_tW)

# Function to compute the ultimate ruin probability
def psi_selfish_u(u, b, c, p, q, λ):
    denom = 1+2*p-p**2
    p_U = [(1+p*(1-p)+p*(1-p)**2*(1-q))/denom, (p*q*(1-p)**2)/denom, (p**2+p**2*(1-p))/denom]
    def f(x): 
        return(c * x + λ * (p_U[0] + p_U[1] * np.exp(-b * x) + p_U[2] * np.exp(- 2 * b * x) - 1))
    Δ = (λ * b * p_U[1])**2 + 8 * b * λ * c * p_U[2]
    x_min = -np.log((- λ * b * p_U[1] + Δ**(1/2)) / 2 / (2* b * λ * p_U[2]))/b
    theta_star = scipy.optimize.root_scalar(f, bracket=[x_min, 100],\
                                            method='brentq').root

    return(np.exp(-theta_star * u))


# Test
# u, b, c, p, q, λ = 2, 5, 0.4, 0.1, 0.5, 2
# psi_u(u, b, c, p, λ), psi_selfish_u(u, b, c, p, q, λ)

# Function to generate a trajectory of Z_n (keeping track of the number of block in Sam's buffer)
def Z_N_t(n, ξ, Z_0):
    # n is the number of blocks discovered by the network
    # ξ is a sequence of n Bernouilli random variables of parameter p (ξ = np.append(0,np.random.binomial(1, p, size = n)))
    # Z_0 is the initial state of the Markov chain
    
    
    #At the beginning, no blocks in the buffer
    Z = np.zeros(n + 1)
    for k in range(n + 1):
        #Initialization of the buffer
        if k == 0:
            Z[k] = Z_0
        else:
            # if the buffer is empty
            if Z[k-1] == 0:
                Z[k] = Z[k-1] + ξ[k]
            # If the buffer contains 1 block
            elif Z[k-1] == 1:
                #If Sam finds a block
                if ξ[k] == 1:
                    Z[k] = 0
                else:
                    Z[k] = 0.5
            # If the buffer is empty and a fork is ongoing
            elif Z[k-1] == 0.5:
                Z[k] = 0
    return(Z)


# Function that simulate trajectories of the reward process
def U_N_t(n, p, q, Z_0, rg):
    # n is the number of blocks discovered by the network (the value of N_T)
    # Sequence of block discovery
    # Z_0 is the initial state of the Markove chain
    # rg sets the seed of the random number generator
    ξ = np.append(0,rg.binomial(1, p, size = n))
    # Sequence of miner's choice
    ζ = np.append(0, rg.binomial(1, q, size = n))
    # Sequence of buffer values 
    Z = Z_N_t(n, ξ, Z_0)
    U = np.zeros(n+2)
    for k in range(n+1):
        if k == 0:
            U[k] = 0
        else:
            # If the buffer is empty, 
            if Z[k-1] == 0:
                U[k] = 0
            # If the buffer contains one block
            elif Z[k-1] == 1:
                # If the others find a block  
                if ξ[k] == 0:
                    #No reward because the chain is forked 
                    U[k] = 0
                # If Sam finds a block  
                else:
                    #Reward for two blocks 
                    U[k] = 2
            # If a fork is ongoing
            else:
                # If Sam finds a block
                if ξ[k] == 1:
                    #Then reward for two blocks
                    U[k] = 2
                # If the others find a block 
                else:
                    # If they append it to their branch
                    if ζ[k] == 0:
                        #No reward
                        U[k] = 0
                    # If they append it to Sam's branch
                    else:
                        #Reward for one block
                        U[k] = 1    
    return(U.astype(int))

# Test
# n, p, q, Z_0, rg = 5, 0.6, 0.8, 0, Generator(PCG64(12))
# print(U_N_t(n, p, q, Z_0, rg))


# Function that produce simulate data for exponential time horizon
def sim_data_self_T(t, λ, p, q, Z_0, K, rg):
    T_exp = rg.exponential(t, K)
    N_t = np.concatenate([rg.poisson(λ * T_exp[k], 1) for k in range(K)])
    T = [np.append(np.append(0,np.sort(np.random.uniform(0, T_exp[k], N_t[k]))),T_exp[k]) for k in range(K)]
    U = [U_N_t(N_t[k], p, q, Z_0, rg) for k in range(K)]
    return({'T_exp':T_exp, 'N_t':N_t, 'T':T, 'U':U})


# Function to compute the revenue function over an exponential time horiszon via MC simulations
def V_T_self_MC(u, t, λ, p, q, b, c, sim_data):
         
    T_exp = sim_data['T_exp']
    K = len(T_exp)
    # Number of jumps up to time t
    N_t = sim_data['N_t']
    # Jump times when R_t > 0
    T = sim_data['T'] 
    # Reward associated to each jump
    U = sim_data['U']
    # Surplus at the end of the time horizon
    R_t = np.array([u - c * T_exp[k] + b * sum(U[k]) for k in range(K)])
    # Surplus at check times
    R_s = [np.array([u - c * T[k][i] + b*sum(U[k][0:i]) for i in range(N_t[k] + 2)]) for k in range(K)]

    no_ruin = np.array([np.all(R_s[k] > 0) for k in range(K)])
    ψ = 1 - np.mean(no_ruin)
    return(np.array([ψ, np.sqrt(ψ*(1-ψ)/K)*norm.ppf(0.975), np.mean(R_t * no_ruin.astype('uint8')),\
                     np.std(R_t * no_ruin.astype('uint8'))/np.sqrt(K) * stu.ppf(0.975,K-1)]))

# Test
# Parameter operational cost
# pkW, network_yearly_tW, pBTC, nBTC = 0.04, 77.78, 9938, 12.5
# u, t, λ, p, q,Z_0, b = 10000, 6, 6, 0.1, 0.5,0, pBTC * nBTC
# rg = Generator(PCG64(12))
# c, sim_data = p * pkW * network_yearly_tW*10**9 / 365 / 24, sim_data_self_T(t, λ, p, q, Z_0, 3, rg)
# η(p, q, b, λ, pkW, network_yearly_tW)
# V_T_self_MC(u, t, λ, p, q, b, c, sim_data)
# V_u_T_MC(u, t, b, c, p, q, λ, Z_0, 100000,rg)

# Function to compute the revenue when doing selfish moning
def V_u_T_MC(u, t, b, c, p, q, λ, Z_0, K,rg):
    # Z_0 is the initial state of the Markov chain 
    # K is the number of trajectories
    T_exp = np.random.exponential(t, K)
    # Number of blocks generated
    N_t = np.array([np.random.poisson(λ * T_exp[k], 1)[0] for k in range(K)])

    # Jump times when N_t>0 
    T = [np.append(np.append(0,np.sort(np.random.uniform(0, T_exp[k], N_t[k]))), T_exp[k]) for k in range(K)]

    #labeling the blocks 1 =sam, 0 = Others
    U = [U_N_t(N_t[k], p, q, Z_0, rg) for k in range(K)]

    R_t = np.array([u - c * T_exp[k] + b * sum(U[k]) for k in range(K)])
    # Surplus at check times
    R_s = [np.array([u - c * T[k][i] + b*sum(U[k][0:i]) for i in range(N_t[k] + 2)]) 
           for k in range(K)]

    no_ruin = np.array([np.all(R_s[k] > 0) for k in range(K)])

    expected_surplus = np.dot(no_ruin, R_t) / K
    
    return(expected_surplus)

# Test
# u, t, b, c, p, q,  λ, Z_0, K = 15, 20, 3, 0.2, 0.1, 0.5, 2,0, 1000
# V_u_T_MC(u, t, b, c, p, q, λ, Z_0, K)
