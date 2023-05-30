import os, time

import numpy as np
import torch.optim as optim
import torch

from tqdm import tqdm
from utils import perdictive_ability

def train_model(model, train_dataloader, test_dataloader, save_path, device, lr, epoch, h2):

   
    model._init_weights()
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    log = open(os.path.join(save_path,'log.txt'),'w')
    log.close()
    
    s_time = time.time()
    for e in range(epoch):
        for i, data in enumerate(train_dataloader):

            # forward
            X = data['X'].to(device)
            true_y = data['y'].to(device)
            pred_y = model(X) 

            pa = perdictive_ability(pred_y, true_y, h2)
            loss = (pred_y - true_y).abs().mean()

            monitor = f'epoch {e+1} ({i+1}/{len(train_dataloader)}): loss ({round(float(loss),3)}), p.a ({round(float(pa),3)})'
            print(monitor+' '*10,end='\r')

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(monitor+' '*10)

        log = open(os.path.join(save_path,'log.txt'),'a')
        log.write(f'{monitor}\n')
        log.close()

    running_time = time.time() - s_time

    # save last parameters and load best paramters
    torch.save(model.state_dict(), os.path.join(save_path,f'last.pth'))
    del loss, optimizer
    return running_time



def test_model(model, test_dataloader, device, save_path):


    model.eval()
    
    s_time = time.time()
    pred_y = []
    ids = []
    print('Testing...')
    for data in tqdm(test_dataloader):
        X = data['X'].to(device)
        a, d, e  =  data['a'].to(device), data['d'].to(device), data['e'].to(device)
        with torch.no_grad():
            batch_pred_y = model(X)
        pred_y.append(batch_pred_y +a +d +e)
        ids.append(data['id'])

    pred_y = torch.cat(pred_y)
    ids = np.concatenate(ids)

    running_time = time.time() - s_time


    with open(os.path.join(save_path,f'sol.txt'), 'w') as save:
        save.write('IDS\tTRUE\tdeepGBLUP\n')
        for id,  py in zip(ids, pred_y):
            save.write(f'{id}\t{py}\n')
    return running_time

