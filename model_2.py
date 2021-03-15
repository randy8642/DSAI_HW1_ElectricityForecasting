import torch.nn as nn
import torch

#%% M
class model_01(nn.Module):
    def __init__(self, H, in_sz):
        super(model_01, self).__init__()
        tot_len = H*in_sz
        self.FC = nn.Sequential(
            nn.Linear(tot_len, 1024),
            nn.Tanh(),
            nn.Linear(1024, 64),
            nn.Tanh(),            
            nn.Linear(64, 1)
            )
    def forward(self, x):
        bz = x.size(0)
        y = self.FC(x.reshape(bz, -1))
        return y.view(bz, -1)

class model_02(nn.Module):
    def __init__(self, H, in_sz):
        super(model_02, self).__init__()
        self.GRU = nn.GRU(in_sz, 1024, 2, bidirectional=False)  
        tot_len = H*1024
        self.FC = nn.Sequential(
            nn.Linear(tot_len, 4096),
            nn.Tanh(),
            nn.Linear(4096, 256),
            nn.Tanh(),            
            nn.Linear(256, 1)
            )
    def forward(self, x):
        bz = x.size(0)
        x_GRU, hn = self.GRU(x)
        y = self.FC(x_GRU.reshape(bz, -1))
        return y.view(bz, -1)
        

#%% Test
if __name__ == '__main__':
    IN = torch.rand(32,30,464)
    F = model_01(30,464)
    Y = F(IN)
    print("Pred >>", Y.size())