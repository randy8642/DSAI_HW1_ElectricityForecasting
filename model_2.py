import torch.nn as nn
import torch

#%% M
class model_01(nn.Module):
    def __init__(self, H, in_sz):
        super(model_01, self).__init__()
        tot_len = H*in_sz
        self.FC = nn.Sequential(
            nn.Linear(tot_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),            
            nn.Linear(256, 64),
            nn.ReLU(),      
            nn.Linear(64, 1)
            )
    def forward(self, x):
        bz = x.size(0)
        y = self.FC(x.reshape(bz, -1))
        return y.view(bz, -1)
        

#%% Test
if __name__ == '__main__':
    IN = torch.rand(32,30,464)
    F = model_01(30,464)
    Y = F(IN)
    print("Pred >>", Y.size())