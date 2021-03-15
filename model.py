import torch.nn as nn
import torch

#%% M
class model_01(nn.Module):
    def __init__(self, H, in_sz):
        super(model_01, self).__init__()
        tot_len = H*in_sz
        self.FC = nn.Sequential(
            nn.Linear(tot_len, tot_len//16),
            nn.LayerNorm(tot_len//16),
            nn.ReLU(),
            nn.Linear(tot_len//16, tot_len//128),
            nn.LayerNorm(tot_len//128),
            nn.ReLU(),            
            nn.Linear(tot_len//128, 30)
            )
    def forward(self, x):
        bz = x.size(0)
        y = self.FC(x.reshape(bz, -1))
        return y.view(bz, -1)

class model_02(nn.Module):
    def __init__(self, in_sz, out_sz, H):
        super(model_02, self).__init__()
        self.GRU = nn.GRU(in_sz, out_sz, 2, batch_first=True)
        self.FC = nn.Sequential(
            nn.Linear(out_sz*H, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 30)
            )
    def forward(self, x):
        bz = x.size(0)
        x_GRU, hn = self.GRU(x)
        pred = self.FC(x_GRU.reshape(bz, -1))
        return pred.reshape(bz, -1)
        

#%% Test
if __name__ == '__main__':
    IN = torch.rand(32,30,772)
    F = model_01(30,772)
    Y = F(IN)
    print("Pred >>", Y.size())
