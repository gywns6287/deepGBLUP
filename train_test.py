import random
import os
from tqdm import tqdm

import numpy as np
import torch.optim as optim
import torch

from utils import perdictive_ability
from model import  deepGBLUP

def lr_epoch_search(train_X, train_y, lr, epoch, h2, device, save_path, grid_search, vali_split):

    # split data to train and vali
    split = int(len(train_X) * (1-vali_split))
    inds = list(range(len(train_X)))
    random.shuffle(inds)

    vali_X = train_X[inds[split:]]
    vali_y = train_y[inds[split:]]
    train_X = train_X[inds[:split]]
    train_y = train_y[inds[:split]].to(device)
    X = torch.cat([train_X, vali_X], dim=0).to(device)

    # make model
    model = deepGBLUP(train_y, train_X.shape[0], train_X.shape[0] + vali_X.shape[0], train_X.shape[1], h2).to(device)

    # lr and epoch searching
    best_lr = lr[0]
    best_monitor = -1
    best_epoch = 0

    # searching the best init weights.
    print('Searching the best init weights and epoch.., the model will be save at "'+os.path.join(save_path,f'init.pth')+'"')
    for e in tqdm(range(200)):
        model._init_weights()
        pred_y, _ =  model(X)
        monitor = perdictive_ability(pred_y[len(train_y):].detach().cpu(), vali_y, h2)
        if monitor > best_monitor:
            best_monitor = monitor
            torch.save(model.state_dict(), os.path.join(save_path,f'init.pth'))

    if not grid_search:
        return lr[0], epoch 

    # searching the best learing rate and epoch
    print('Searching the best learning rate and epoch..')
    for l in lr:
        model.load_state_dict(torch.load(os.path.join(save_path,f'init.pth')), strict=False)
        best_epoch_per_lr = 0
        best_monitor_per_lr = 0
        optimizer = optim.AdamW(model.parameters(), lr=l)

        print('-'*100)
        print(f'For the learning rate {l}...')
        for e in range(epoch):

            # feedforward
            pred_y, _ = model(X)
            monitor = perdictive_ability(pred_y[len(train_y):].detach().cpu(), vali_y, h2)
            loss = perdictive_ability_loss(pred_y[:len(train_y)], train_y)
            print(f'epoch {e+1}: loss: {loss.detach()}, predictive ability: {monitor}')

            # save best epoch
            if monitor > best_monitor_per_lr:
                best_epoch_per_lr = e + 1
                best_monitor_per_lr = monitor

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Learining rate {l}: best_epoch {best_epoch_per_lr}, predictive ability: {best_monitor_per_lr}')

        if best_monitor_per_lr > best_monitor:
            best_epoch = best_epoch_per_lr
            best_lr = l
            best_monitor = best_monitor_per_lr

    print('-'*100)
    print(f'Learing rate {best_lr} and epoch {best_epoch} are selected with predictive ability:{best_monitor} for training.')
    return best_lr, best_epoch

def train_model(model, train_X, train_y, test_X, lr, epoch, h2, device, save_path):
    print('-'* 100)
    print('Training deepGBLUP with the best learning rate and epoch..')

    # Load data to device
    train_y = train_y.to(device)
    X = torch.cat([train_X, test_X], dim=0).to(device)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    log = open(os.path.join(save_path,'log.txt'),'w')
    log.close()
    
    for e in range(epoch):
        pred_y, _ = model(X)

        loss = perdictive_ability_loss(pred_y[:len(train_y)], train_y)
        monitor = perdictive_ability(pred_y[:len(train_y)].detach().cpu(), train_y.detach().cpu(), h2)
        print(f'epoch {e+1}: loss: {loss.detach()}, predictive ability: {monitor}')

        log = open(os.path.join(save_path,'log.txt'),'a')
        log.write(f'epoch {e+1} loss: {loss.detach()}, predictive ability: {monitor}\n')
        log.close()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(save_path,f'last.pth'))

def test_model(model, train_ids, test_ids, train_X, test_X, h2, device, save_path):
    X = torch.cat([train_X, test_X], dim=0).to(device)
    ids = np.concatenate([train_ids, test_ids],axis=0) 

    model.eval()
    pred_y, _ = model(X)
    pred_y = pred_y.detach().cpu()

    with open(os.path.join(save_path,f'sol.txt'), 'w') as save:
        save.write('IDS\tgEBV\n')
        for i, g in zip(ids, pred_y):
            save.write(f'{i}\t{g}\n')


def perdictive_ability_loss(true, pred):
    pred_mean = pred.mean()
    true_mean = true.mean()

    f1 = torch.sum((pred - pred_mean) * (true - true_mean))
    f2 = torch.sqrt(torch.sum((pred - pred_mean)**2) * torch.sum((true - true_mean)**2))

    cor = f1/f2 
    return 1-cor



