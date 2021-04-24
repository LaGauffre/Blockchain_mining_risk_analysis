# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:11:50 2020

@author: Pierre-O
"""
# Function to compute the ruin probabilities 
# and the revenue function for a miner who is 
# following the protocol 
import numpy as np
import scipy as sc
import pandas as pd
import math as ma
from numpy.random import Generator, PCG64, SeedSequence
from scipy.special import binom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import t as stu

from scipy.optimize import fsolve
import scipy.optimize
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

# The functions herafter tends to share the following
# parameters:
# - u is the initial reserve
# - t is the time horizon
# - b is the reward for finding a block
# - c is the operational cost
# - p is the share of computing power
# - λ is the block arrival intensity
# - rg = Generator(PCG64(123)) sets the random seed

# rg = Generator(PCG64(123))
# p, λ, K, t = 0.16, 6, 10, 3
# Function that produce simulate data for deterministic time horizon
def sim_data_protocol_t(λ, p, t, K, rg):
    N_t = rg.poisson(p * λ * t, K)
    T = np.array([np.sort(rg.uniform(0, t, N_t[k])) for k in range(K)])
    return({'N_t':N_t, 'T':T})



# Function to compute the finite time ruin probability and the revenue function 
# via MC simukations
def ψ_V_t_MC(u, t, b, c, p, λ, sim_data):  
    # Number of jumps up to time 
    N_t = sim_data['N_t']
    # Number of trajectories
    K = len(N_t)
    # Surplus at terminal time t
    R_t = np.array([max(u + N_t[k] * b - c* t , 0) for k in range(K)])
    # Jump times when R_t > 0
    T = sim_data['T'] 
    # Check points time 
    S = [np.array([(u + b * n) / c  for n in range(N_t[k])]) for k in range(K)]
    # Ruin or not
    no_ruin = np.array([ (all(T[k] < S[k]))& (R_t[k] > 0)  for k in range(K)])
    ψ = 1 - np.mean(no_ruin)
    return(np.array([ψ, np.sqrt(ψ*(1-ψ)/K)*norm.ppf(0.975), np.mean(R_t * no_ruin.astype('uint8')),\
                     np.std(R_t * no_ruin.astype('uint8'))/np.sqrt(K) * stu.ppf(0.975,K-1)]))

# u = 10000
# # Share of the computing power
# p = 0.16
# # $/kW
# pkW= 0.06
# # Network yearly consumption in kW
# network_yearly_kW = 70 * 10**9
# # operational cost per hour
# c = p * pkW * network_yearly_kW / 365 / 24
# #Reward fo mining a block  in dollars
# b = 12.5* 6943.7
# rg = Generator(PCG64(12))
# p, λ, K, t = 0.16, 6, 100, 3
# sim_data = sim_data_protocol_t(λ, p, t, K, rg)

# ψ_V_t_MC(u, t, b, c, p, λ, sim_data)

# Function that produce simulate data for exponential time horizon
def sim_data_protocol_T(λ, p, t, K, rg):
    T_exp = rg.exponential(t, K)
    N_t = np.concatenate([rg.poisson(p * λ * T_exp[k], 1) for k in range(K)])
    T = np.array([np.sort(rg.uniform(0, T_exp[k], N_t[k])) for k in range(K)])
    return({'T_exp':T_exp, 'N_t':N_t, 'T':T})

# rg = Generator(PCG64(12))
# p, λ, K, t = 0.16, 6, 10, 3
# sim_data_protocol_T(λ, p, t, K, rg)

# Function to compute the revenue function over an exponential time horiszon via MC simulations
def ψ_V_T_MC(u, t, b, c, p, λ, sim_data):
    # K is the number of trajectories simulated         
    T_exp = sim_data['T_exp']
    K = len(T_exp)
    # Number of jumps up to time t
    N_t = sim_data['N_t']
    # Surplus at terminal time t
    R_t = np.array([max(u + N_t[k] * b - c* T_exp[k] , 0) for k in range(K)])
    # Jump times when R_t > 0
    T = sim_data['T'] 
    # Check points time 
    S = [np.array([(u + b * n ) / c  for n in range(N_t[k])]) for k in range(K)]
    # Ruin or not
    no_ruin = np.array([ (all(T[k] < S[k]))& (R_t[k] > 0)  for k in range(K)])
    ψ = 1 - np.mean(no_ruin)
    return(np.array([ψ, np.sqrt(ψ*(1-ψ)/K)*norm.ppf(0.975), np.mean(R_t * no_ruin.astype('uint8')),\
                     np.std(R_t * no_ruin.astype('uint8'))/np.sqrt(K) * stu.ppf(0.975,K-1)]))

# rg = Generator(PCG64(12))
# p, λ, K, t = 0.16, 6, 100000, 3
# sim_data = sim_data_protocol_T(λ, p, t, K, rg)
# u = 1000
# # Share of the computing power
# p = 0.16
# # $/kW
# pkW= 0.06
# # Network yearly consumption in kW
# network_yearly_kW = 70 * 10**9
# # operational cost per hour
# c = p * pkW * network_yearly_kW / 365 / 24
# #Reward fo mining a block  in dollars
# b = 12.5* 6943.7
# ψ_V_t_MC(u, t, b, c, p, λ, sim_data)
# V_u_T(u, t, b, c, p, λ)

def ψ_t(u, t, b, c, p, λ):
    n = ma.floor((c * t - u) / b)
    return(sum([u / (u + b * k) * poisson.pmf(k, p * λ * (u + b * k) / c) for k in range(n+1)]))

# #  Test
# u, t, b, c, p, λ, K = 2, 20, 1, 0.4, 0.1, 2,  10000
# # print(u - c*t + lam * p * b)
# psi_MC = np.array([psi_u_t_MC(u, t, b, c, p, λ, 1000) for i in range(1000)])
# result = plt.hist(psi_MC, bins=20, color='c', edgecolor='k', alpha=0.65)
# plt.axvline(psi_u_t(u, t, b, c, p, λ), color='k', linestyle='dashed', linewidth=1)
# plt.show()

# Function to compute the ultimate ruin probability
def ψ(u, b, c, p, λ):
    def f(x): 
        return(c * x + p * λ * (np.exp(-b*x) - 1))
    theta_star = scipy.optimize.root_scalar(f, bracket=[np.log(p * b * λ / c) / b, 100],
                                            method='brentq').root
    return(np.exp(-theta_star * u))

# # Model parameters
# u, b, c, p, λ = 2, 5, 0.4, 0.1, 2
# psi_u(u, b, c, p, λ),psi_u_t(u,1000, b, c, p, λ)

# Function to compute the revenue function over a deterministic time horizon via MC simulations
def V_t_MC(u, t, b, c, p, λ, K):
    # K is the number of trajectories simulated     
    # Number of jumps up to time t
    N_t = [np.random.poisson(p * λ * t, 1) for k in range(K)]
    # Surplus at terminal time t
    R_t = np.array([max(u + N_t[k] * b - c* t , 0) for k in range(K)])
    # Jump times when R_t > 0
    T = [np.sort(np.random.uniform(0, t, N_t[k])) for k in np.where(R_t > 0)[0]]
    # Check points time 
    S = [np.array([(u + b * n ) / c  for n in range(N_t[k][0])]) for k in np.where(R_t > 0)[0]]
    # Ruin or not
    no_ruin = np.array([all(T[n] < S[n])  for n in range(np.size(np.where(R_t > 0)[0]))])
    return(np.dot(no_ruin, R_t[R_t > 0]) / K)

# Function to compute the revenue function over a finite time horizon using Prop 1
def V_t(u, t, b, c, p, λ, K):
    # K is a truncation order for the infinite serie     
    U = [min(1, (u + b * k) / c / t) for k in range(K)]
    V = [max(0, u + k * b - c * t) * (-1) ** k * poisson.pmf(k, p * λ * t) for k in range(K)]
    G_k = [1]
    for k in range(1,K,1):
        G_k.append(
            -sum([
                binom(k,i) * U[i]**(k-i)*G_k[i] 
                for i in range(0,k,1)]))
    return(np.dot(V,G_k))

# # Test
# u, t, b, c, p, λ = 20, 10, 5, 0.3, 0.1, 1
# V_MC = np.array([V_u_t_MC( u, t, b, c, p, λ, 1000) for i in range(1000)])
# result = plt.hist(V_MC, bins=20, color='c', edgecolor='k', alpha=0.65)
# plt.axvline(V_u_t(u, t, b, c, p, λ, 30), color='k', linestyle='dashed', linewidth=1)
# plt.axvline(np.mean(V_MC), color='r', linestyle='dashed', linewidth=1)
# plt.show()
# np.mean(V_MC),V_u_t(u, t, b, c, p, λ, 30)

# Function to compute the revenue function over an exponential time horiszon via MC simulations
def V_T_MC(u, t, b, c, p, λ, K):
    # K is the number of trajectories simulated         
    T_exp = np.random.exponential(t, K)
    # Number of jumps up to time t
    N_t = [np.random.poisson(p * λ * T_exp[k], 1) for k in range(K)]
    # Surplus at terminal time t
    R_t = np.array([max(u + N_t[k] * b - c* T_exp[k] , 0) for k in range(K)])
    # Jump times when R_t > 0
    T = [np.sort(np.random.uniform(0, T_exp[k], N_t[k])) for k in np.where(R_t > 0)[0]]
    # Check points time 
    S = [np.array([(u + b * n ) / c  for n in range(N_t[k][0])]) for k in np.where(R_t > 0)[0]]
    # Ruin or not
    no_ruin = np.array([all(T[n] < S[n])  for n in range(np.size(np.where(R_t > 0)[0]))])
    return(np.dot(no_ruin, R_t[R_t > 0]) / K)

# Function to compute the value function over an exponential time horizon via Prop 2
def V_T(u, t, b, c, p, λ):
    γ = p * λ * b - c
    def f(x): 
            return(c * t * x + (1 + p * λ * t) - p *λ * t * np.exp(x * b) )
    rho_star = scipy.optimize.root_scalar(f, bracket=[np.log(c / p * b * λ) / b-10, 
                                                      np.log(c / p / b / λ) / b],
                                          method='brentq').root
    return(γ * t + u - γ * t * np.exp(rho_star * u))

# Function to compute the ruin probability over an exponential time horizon via cor 1
def ψ_T(u, t, b, c, p, λ):
    
    def f(x): 
            return(c * t * x + (1 + p * λ * t) - p *λ * t * np.exp(x * b) )
    rho_star = scipy.optimize.root_scalar(f, bracket=[np.log(c / p * b * λ) / b-10, 
                                                      np.log(c / p / b / λ) / b],
                                          method='brentq').root
    return(np.exp(rho_star * u))

# Function to compute the revenue function over an exponential time horizon via Prop 2
def V_T_bis(u, t, b, p, λ, pkW, W):
    c = p * pkW * W
    γ = p * λ * b - c
    def f(x): 
            return(c * t * x + (1 + p * λ * t) - p *λ * t * np.exp(x * b) )
    rho_star = scipy.optimize.root_scalar(f, bracket=[np.log(c / p * b * λ) / b-10, 
                                                      np.log(c / p / b / λ) / b],
                                          method='brentq').root
    return(γ * t + u - γ * t * np.exp(rho_star * u))

def ψ_T_bis(u, t, b, p, λ, pkW, W):
    c = p * pkW * W
    def f(x): 
            return(c * t * x + (1 + p * λ * t) - p *λ * t * np.exp(x * b) )
    rho_star = scipy.optimize.root_scalar(f, bracket=[np.log(c / p * b * λ) / b-10, 
                                                      np.log(c / p / b / λ) / b],
                                          method='brentq').root
    return(np.exp(rho_star * u))


# # Test
u, t, b, p, λ, pkW, network_yearly_tW = 200000, 28 * 14,12.5* 6943.7, 0.2, 6, 0.02, 77.78 
# V_MC = np.array([V_u_t_MC( u, t, b, c, p, λ, 1000) for i in range(1000)])
# result = plt.hist(V_MC, bins=20, color='c', edgecolor='k', alpha=0.65)
# plt.axvline(V_u_t(u, t, b, c, p, λ, 30), color='k', linestyle='dashed', linewidth=1)
# plt.axvline(np.mean(V_MC), color='r', linestyle='dashed', linewidth=1)
# plt.show()
# np.mean(V_MC),V_u_t(u, t, b, c, p, λ, 30)

# Model parameters
# u, t, b, c, p, λ = 5, 20, 5, 0.3, 0.1, 1
# V_MC = np.array([V_u_T_MC(u, t, b, c, p, λ, K) for i in range(1000)])
# result = plt.hist(V_MC, bins=20, color='c', edgecolor='k', alpha=0.65)
# plt.axvline(V_u_T(u, t, b, c, p, λ), color='k', linestyle='dashed', linewidth=1)
# plt.axvline(np.mean(V_MC), color='r', linestyle='dashed', linewidth=1)
# plt.show()
