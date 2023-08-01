
# deepGBLUP: Integration of deep learning and GBLUP for accurate genomic prediction.
 

## Model summary
This repo is the official Code of deepGBLUP: joint deep learning networks and GBLUP framework for accurate genomic prediction of complex traits in Korean native cattle. For model details, please refer to the paper [[link]](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-023-00825-y).

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
lr: learning rate
epoch: epoch
batch_size: batch_size
device: type 'cpu' if you use cpu device, or type 'cuda' if you use gpu device.
h2: heritability

# save config
save_path: path to save results
```
2. Run deepGBLUP
```
python main.py
```

3. Output files
```
last.pth: trained weights file
log.txt: training log file
setting.txt: text file to save configuration
sol.txt: text file to save individuals' gEBV.

```
