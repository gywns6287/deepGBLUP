import os
from tqdm import tqdm
from math import sqrt, log10
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def perdictive_ability(true, pred, h2):
    pred_mean = pred.mean()
    true_mean = true.mean()

    f1 = torch.sum((pred - pred_mean) * (true - true_mean))
    f2 = torch.sqrt(torch.sum((pred - pred_mean)**2) * torch.sum((true - true_mean)**2))

    cor = f1/f2 
    return float(cor/sqrt(h2))

def association_study(model, train_X, test_X, y, device, SNP_names, bim_path, save_path):
    X = torch.cat([train_X, test_X], dim=0).to(device)

    # Extract LD-effects interfused SNPs
    _, M = model(X.to(device))
    M = M[:len(y)].detach().cpu().numpy()
    y = y.numpy()

    effects = []
    print('Doing association study')
    for i in tqdm(range(M.shape[1])):
        slope, intercept, r_value, p_value, std_err = stats.linregress(y, M[:,i])
        e = -log10(p_value)
        effects.append(e)
    effects = np.array(effects)

    if bim_path:
        # call marker information
        bim ={}
        with open(bim_path) as b:
            for line in b:
                line_ = line.split()
                if '_' in line_[1]:
                    line_[1] = '_'.join(i.split('_')[:-1])
                bim[line_[1]] = (line_[0],line_[3])

    # save effect as txt file    
    save =  open(os.path.join(save_path,'snp_effects.txt'), 'w')

    save.write('Marker\tChr\tPos\tEffect\n')
    chr_count = []
    for m, e in zip(SNP_names, effects):
        if '_' in m:
            m = '_'.join(m.split('_')[:-1])
        if bim_path:
            c, p = bim[m]
            chr_count.append(c)
        else:
            c = p = 'na'
        save.write(f'{m}\t{c}\t{p}\t{e}\n')
    save.close()

    # vis effect as manhattan plot
    # calculate chromosom axis position
    if bim_path:
        chr_count = Counter(chr_count)
        chrs =  sorted(chr_count.keys(), key = lambda k: int(k))
        chr_idx = []
        cumsum = 0
        for c in chrs:
            chr_idx.append(cumsum + chr_count[c]//2)
            cumsum += chr_count[c]

    # vis
    fig, ax = plt.subplots(figsize=(13,5))
    ylim = (0, np.max(effects)*1.1)

    ax.set_title('SNP effects')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True,ls='--',lw=1,alpha=1,axis='x')
    ax.set_ylim(ylim) 
    if bim_path:
        plt.xticks(chr_idx,chrs)

    # scatter plot
    cumsum = 0
    colors = ['tab:blue','tab:red']
    if bim_path:
        for c in chrs:
            chr_len = chr_count[c]
            chr_which = range(cumsum,cumsum+chr_len)
            chr_effects = effects[chr_which]
            ax.scatter(chr_which, chr_effects, s=5, c = colors[int(c)%2])
            cumsum += chr_len
    else:
        ax.scatter(list(range(len(effects))), effects, s=5, c = colors[0])

    plt.savefig(os.path.join(save_path,f'manhatan.png'), bbox_inches='tight', pad_inches=0)


