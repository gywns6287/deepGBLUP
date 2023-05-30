import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LocalLinear(nn.Module):
    def __init__(self,in_features,local_features,kernel_size,stride=1,bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - 1

        fold_num = (in_features+self.padding -self.kernel_size)//self.stride+1
        self.weight = nn.Parameter(torch.randn(fold_num,kernel_size,local_features))
        self.bias = nn.Parameter(torch.randn(fold_num,local_features)) if bias else None

        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x:torch.Tensor):
        x = F.pad(x,[0, self.padding],value=0)
        x = x.unfold(-1,size=self.kernel_size,step=self.stride)
        x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)+self.bias
        return x.squeeze(2)

def matrix_normalization(T):
    T[range(len(T)),range(len(T))] = 0
    D = np.diag(np.sum(T,axis=0) ** (-1/2))
    return np.matmul(np.matmul(D,T),D)

class deepGBLUP(nn.Module):
    def __init__(self, ymean, num_snp):
        super(deepGBLUP, self).__init__()

        # set mean
        self.mean = ymean

        # # set LCL
        self.encoder = nn.Sequential(
            LocalLinear(num_snp, 1, kernel_size=5,stride=1),
            nn.LayerNorm(num_snp),
            nn.GELU(),
            LocalLinear(num_snp, 1, kernel_size=3,stride=1),
            # nn.Sigmoid()
        )         
      
        self.bias = nn.Linear(num_snp,1)
        # self.encoder = nn.DataParallel(self.encoder,device_ids=['cuda:0','cuda:1'])

        self._init_weights()
        
    def _init_weights(self):
        # weight init
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self,X):
        # bias
        X = self.encoder(X) + X
        b = self.bias(X).squeeze(1) 
        

        return self.mean + b

if __name__ == '__main__':
    x = torch.rand(16, 44, 1024)
    m = GPNet(45056, ld_size=1024)
    o = m(x)
    print(o.shape)
