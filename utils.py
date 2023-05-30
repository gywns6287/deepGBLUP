import torch
import math
import numpy as np
import os

from tqdm import tqdm


def mixed_model(Z, A, lamb, y):
        y = y - y.mean()
        e  = torch.inverse(Z.T @ Z + A *lamb) @ Z.T @ y
        return e

def call_Z(ref_len, whole_len):
    Z = torch.zeros((ref_len, whole_len),dtype=torch.float32)
    for i in range(ref_len): Z[i,i] = 1
    return Z

def make_G_D_E(X, invers=True, device='cpu'):

    # cal freq matrix
    n,k = X.shape
    pi = X.sum(0)/(2*n) # X.mean(0)/2
    P = (pi).unsqueeze(0) 

    # make A (dummay pedigree)
    A = torch.eye(len(X))

    # make G
    Z = X - 2*P 
    G = (Z @Z.T)  /(2*(pi*(1-pi)).sum()) 

    # make D
    print('make dominance matrix')
    W = X.clone()
    for j in tqdm(range(W.shape[1])):
        W_j = W[:,j]
        W_j[W_j == 0] = -2*(pi[j]**2)
        W_j[W_j == 1] = 2*(pi[j]*(1-pi[j]))
        W_j[W_j == 2] = -2*((1-pi[j])**2)

    D = (W @W.T)  /(((2*pi*(1-pi))**2).sum()) 

    # make E
    print('make interaction marker')
    M = X - 1 
    E = 0.5*((M @ M.T) * (M @ M.T))  - 0.5*((M * M) @ (M * M).T) 
    E = E/(torch.trace(E)/n) 
    # E = D
    del W, M

    # rescaling with dummy A
    G = G * 0.99 + A * 0.01
    D = D * 0.99 + A * 0.01
    E = E * 0.99 + A * 0.01

    if invers:
        return torch.inverse(G), torch.inverse(D), torch.inverse(E)
    return G, D, E



def GBLUP(train_X, test_X, train_y, test_y, h2):

    X = torch.cat([train_X, test_X], dim=0)
    train_len = len(train_y)

    # Compute G
    M = X - 1
    pi = X.mean(0)/2
    P = 2*(pi-0.5)
    M = M - P 
    G = (M @M.T)/(2*(pi*(1-pi)).sum())

    G_inv = torch.inverse(G)
    Z = torch.zeros((train_len, len(G_inv)),dtype=torch.float32)
    for i in range(train_len): Z[i][i] = 1
    y_c = (train_y - train_y.mean()).unsqueeze(1)
    y_c = train_y.unsqueeze(1)
    lamb = h2/(1 - h2)
    
    y_gblup = torch.inverse(Z.T @ Z + G_inv*lamb) @ Z.T @ y_c
    # m1 = perdictive_ability(test_y, y_gblup[train_len:][:,0], h2)


    return y_gblup[:,0][:train_len], y_gblup[:,0][train_len:], Z


def perdictive_ability(true, pred, h2=1):
    pred_mean = pred.mean()
    true_mean = true.mean()

    f1 = torch.sum((pred - pred_mean) * (true - true_mean))
    f2 = torch.sqrt(torch.sum((pred - pred_mean)**2) * torch.sum((true - true_mean)**2))
    if f2 == 0:
        return 0

    cor = f1/f2 
    return float(cor/math.sqrt(h2))

