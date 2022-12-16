import os
import time

import torch

from train_test import train_model, test_model, lr_epoch_search
from model import deepGBLUP
from dataset import load_dataset
from utils import association_study

################ CONFIG ####################
# data path
raw_path = 'data/1000_samples.raw' # path of raw file
phen_path = 'data/1000_samples.phen' # path of phenotype file
bim_path = 'data/1000_samples.bim' # optional: path of bim file to save SNP effects. If you don't have bim file just type None 

# train cofig
lr =  [1e-3, 1e-4, 1e-5] # list of cadidate learning rate
epoch = 100 # max value of cadiate epoch
grid_search = True # True: search the best learning rate and epoch; False: just use first lr and max epoch for training.
vali_split = 0.1 # percentage of the validation set in the train set;  

device = 'cuda' # type 'cpu' if you use cpu device, or type 'cuda' if you use gpu device.
h2 = 0.37

# save config
cal_effect = True # True:  Save snp effect with LD-blocks' effect.
save_path = 'outputs' # path to save results

##############################################

####################################################
##                      Caution                   ## 
##  Users unfamiliar with python and pytorch      ##
##  should not modify the code below.             ##
####################################################
os.makedirs(save_path, exist_ok=True)
s_time = time.time()

# Load Dataset
train_X, train_y, train_ids, test_X, test_ids, SNP_names =  load_dataset(raw_path, phen_path)

# Search the best learning rate and epoch
best_lr, best_epoch = lr_epoch_search(train_X, train_y, lr, epoch, h2, device, save_path, grid_search, vali_split)

model = deepGBLUP(train_y, train_X.shape[0], train_X.shape[0] + test_X.shape[0], train_X.shape[1], h2).to(device)
init = torch.load(os.path.join(save_path,f'init.pth'))
del init['y'], init['Z']
model.load_state_dict(init, strict=False)

# Train model
train_model(model, train_X, train_y, test_X, best_lr, best_epoch, h2, device, save_path)

# Test model
test_model(model, train_ids, test_ids, train_X, test_X, h2, device, save_path)
            
if cal_effect:
    # Save snp effect with LD-blocks' effect 
    association_study(model, train_X, test_X, train_y, device, SNP_names, bim_path, save_path)


# Save hyperparameters
e_time = time.time() - s_time
with open(os.path.join(save_path,'setting.txt'),'w') as save:
    save.write(f'Path of raw file: {raw_path}\n')
    save.write(f'Path of phenotype file: {phen_path}\n')
    save.write('-'*50+'\n')
    save.write(f'Best learning rate: {best_lr}\n')
    save.write(f'Best epoch: {best_epoch}\n')
    save.write('-'*50+'\n')
    save.write(f'H2: {h2}\n')
    save.write(f'Running time: {e_time}s\n')
    save.write(f'Device: {device}')