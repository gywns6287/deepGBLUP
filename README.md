
# deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction.
 

## Model summary
We propose a novel genomic prediction algorithm, which integrates deep learning and GBLUP (deepGBLUP). Given SNP markers, the proposed deepGBLUP first extracts epistasis based on locally-connected layers. Then, it estimates initial breeding values through a relation-aware module, which extends GBLUP to the deep learning framework. Finally, deepGBLUP estimates a breeding value bias using a fully connected layer and adds it to the initial breeding value for calculating the final breeding values.

<img src = "https://user-images.githubusercontent.com/71325306/208086095-3471a61a-baf3-4db0-8a42-18f81ebe5842.png" width="30%" height="30%">

## Implementation
### 0. Requirements
We build deepGBLUP on the **Python 3.9**, **Ubuntu 18.04**, and **cuda11.3**. We recommend **anaconda** environment for deepGBLUP.
### 1. Installation
1. Clone this repository
2. Build the virtual environment
```
conda create -n venv python=3.9
conda activate venv
```
3. Install pytorch:
```
# For CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# For CPU Only
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch
```
3. Install requirements
```
pip install -r requirements.txt
```
### 2. Excution
0. Input Data format

**1) raw**

The genotype data of `plink`.  See https://www.cog-genomics.org/plink2/formats#raw for more details.

**2) phenotype**

The phenotype data. It is a `.txt` file involving two columns. First column is animal name, which must match with the one in **raw** file. Second column is phenotype.
deepGBLUP automatically sets the individuals included in **raw** file but not in the **phenotype** file as test individuals.
See data/1000_samples.phen as an example format.

1. Open the 'main.py' file with text editor and set configuration.  You can implement deepGBLUP with sample data by using default configuration.
```
# data path
raw_path: path of raw file
phen_path: path of phenotype file
bim_path (optional): path of bim file to save SNP effects. If you don't have bim file just type None 

# train cofig
lr: list of cadidate learning rate
epoch: max value of cadiate epoch
grid_search: boolean - True: search the best learning rate and epoch; False: just use first lr and max epoch for training.
vali_split: percentage of the validation set in the train set;  
device: type 'cpu' if you use cpu device, or type 'cuda' if you use gpu device.
h2: heritability

# save config
cal_effect:  boolean - True:  Save snp effect with LD-blocks' effect.
save_path: path to save results
```
2. Run deepGBLUP
```
python main.py
```

