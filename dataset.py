import numpy as np 
from tqdm import tqdm
import os

import torch
from torch.utils.data import Dataset
from utils import  make_G_D_E, mixed_model, call_Z


class SNPDataset(Dataset):
    def __init__(self, X, a, d, e, ids, y=None):
        self.X = X
        self.y = y
        self.ids = ids
        self.a = a 
        self.d = d
        self.e = e

    def __len__(self):
        return len(self.ids)
    

    def __getitem__(self,idx):
        if self.y is not None:
            return {
                'X':self.X[idx], 'y':self.y[idx], 
                'a':self.a[idx], 'd':self.d[idx], 'e':self.e[idx],
                'id':self.ids[idx]
            }
        else:
            return {
                'X':self.X[idx],
                'a':self.a[idx], 'd':self.d[idx], 'e':self.e[idx],
                'id':self.ids[idx]
            }

def load_dataset(raw_path, phen_path, h2, device='cpu'):

    # load phen to dict
    train_ids = []
    with open(phen_path) as phen_file:
        phen = {}
        for line in phen_file:
            line_ = line.split()
            phen[line_[0]] = float(line_[1])
            train_ids.append(line_[0])


    # Load snp and Rearrange phen data 
    raw_file = open(raw_path) # SNP
    SNPs = next(raw_file).split()[6:]

    train_X = []
    test_X = []
    train_y = []
    test_ids = []

    for line in tqdm(raw_file):
        line_ = line.split()

        if line_[1] in train_ids:
            train_X.append(line_[6:])        
            train_y.append(phen[line_[1]])
        else:
            test_X.append(line_[6:])        
            test_ids.append(line_[1])

    # to tensor
    train_X = torch.from_numpy(np.array(train_X, dtype=np.float32))
    train_y = torch.tensor(train_y, dtype=torch.float32)
    train_ids = np.array(train_ids,dtype=str)
    test_X = torch.from_numpy(np.array(test_X, dtype=np.float32))
    test_ids = np.array(test_ids,dtype=str)
    
    # cal genetic effects
    X = torch.cat([train_X, test_X], dim=0)
    y = train_y.clone()
    Gi, Di, Ei = make_G_D_E(X, invers=True, device = device)
    Z = call_Z(len(train_X), len(train_X)+len(test_X))
    glamb = (1 - h2)/h2
    dlamb = elamb =  (1 - h2*0.1)/(h2*0.1)
    
    a = mixed_model(Z, Gi, glamb,y)
    d = mixed_model(Z, Di, dlamb,y)
    e = mixed_model(Z, Ei, elamb,y)

    train_a, test_a = a[:len(train_X)], a[len(train_X):]
    train_d, test_d = d[:len(train_X)], e[len(train_X):]
    train_e, test_e = e[:len(train_X)], d[len(train_X):]
    
    train_dataset = SNPDataset(train_X,  train_a, train_d, train_e, train_ids, train_y)
    test_dataset = SNPDataset(test_X, test_a, test_d, test_e, test_ids)
    return train_dataset, test_dataset, X.shape[1], y.mean()



