import numpy as np 
from tqdm import tqdm
import os

import torch

def load_dataset(raw_path, phen_path):
    # load phen to dict
    train_ids = []
    print('Load phen and raw data (it may take a minute.)')
    with open(phen_path) as phen_file:
        phen = {}
        for line in phen_file:
            line_ = line.split()
            phen[line_[0]] = float(line_[1])
            train_ids.append(line_[0])

    # Load SNP data
    raw_file = open(raw_path) # SNP
    SNP_names = next(raw_file).split()[6:]

    train_X = []
    test_X = []
    train_y = []
    test_ids = []
    for line in tqdm(raw_file,total=len(phen)):
        line_ = line.split()
        sequence = line_[6:]
        # fill NA to 0
        sequence = [s if s.lower() != 'na' else '0' for s in sequence]
        if line_[1] in train_ids:
            train_X.append(sequence)   
            train_y.append(phen[line_[1]])
        else:
            test_X.append(sequence)        
            test_ids.append(line_[1])

    # to tensor
    train_X = torch.from_numpy(np.array(train_X, dtype=np.float32))
    train_y = torch.tensor(train_y, dtype=torch.float32)
    train_ids = np.array(train_ids,dtype=str)
    test_X = torch.from_numpy(np.array(test_X, dtype=np.float32))
    test_ids = np.array(test_ids,dtype=str)
    
    return train_X, train_y, train_ids, test_X, test_ids, SNP_names