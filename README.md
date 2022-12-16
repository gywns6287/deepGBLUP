
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
