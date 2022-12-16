import torch
from torch import nn
import torch.nn.functional as F

class LocalLinear(nn.Module):
    def __init__(self,in_features,local_features,kernel_size,stride=1,bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - 1

        fold_num = (in_features+self.padding -self.kernel_size)//self.stride+1
        self.weight = nn.Parameter(torch.randn(fold_num,kernel_size,local_features))
        self.bias = nn.Parameter(torch.randn(fold_num,local_features)) if bias else None

    def forward(self, x:torch.Tensor):
        x = F.pad(x,[0, self.padding],value=0)
        x = x.unfold(-1,size=self.kernel_size,step=self.stride)
        x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)+self.bias
        return x.squeeze(2)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)


class deepGBLUP(nn.Module):
    def __init__(self, y, num_train, num_whole, num_snp, h2):
        super(deepGBLUP, self).__init__()
        self.lamb = nn.Parameter(torch.tensor(h2/(1 - h2)), requires_grad=True)
        self.y = nn.Parameter(y - y.mean(), requires_grad=True)
        Z = torch.zeros((num_train, num_whole),dtype=torch.float32)
        for i in range(num_train): Z[i][i] = 1
        self.Z = nn.Parameter(Z, requires_grad=False)

        self.encoder = nn.Sequential(
            LocalLinear(num_snp, 1, kernel_size=5,stride=1),
            nn.LayerNorm(num_snp),
            nn.GELU(),
            LocalLinear(num_snp, 1, kernel_size=3,stride=1),
            nn.Sigmoid()
        )           
        self.encoder = nn.DataParallel(self.encoder)
        self.scale = nn.Parameter(torch.tensor(1e-5), requires_grad=True)
        self.bias = nn.Linear(num_snp,1)
        self._init_weights()
        
    def _init_weights(self):
        # weight init
        for m in self.modules():
            if isinstance(m, (nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Linear, LocalLinear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, X, b = 8):
        
        X = (X/2)
        M = self.encoder(X) + X

        b = self.bias(M).squeeze(1)
        G = (M @ M.T)
        pi = torch.diagonal(G).unsqueeze(1)
        G = G/pi

        pred_y  = torch.inverse(self.Z.T @ self.Z + torch.inverse(G) *self.lamb) @ self.Z.T @ self.y
        pred_y = pred_y + b * self.scale
        return pred_y, M