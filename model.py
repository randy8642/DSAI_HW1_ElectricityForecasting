import torch.nn as nn
import torch

#%% Attention   
class Temp_AT(nn.Module):
    def __init__(self, fdim):
        super (Temp_AT, self).__init__()
        self.t_at = nn.Sequential(
            nn.Linear(fdim, fdim//4),
            nn.LayerNorm(fdim//4),
            nn.Linear(fdim//4, 1),
            nn.Flatten(),
            nn.Softmax(-1)
            )
    def forward(self, x):
        # IN bz,seq,in_sz

        at_map = self.t_at(x).unsqueeze(-1)
        y = at_map * x
        y = torch.sum(y, dim=1, keepdim=True)
        return y, at_map

#%% M
class model_01(nn.Module):
    def __init__(self, in_sz, out_sz, hid):
        super(model_01, self).__init__()
        self.GRU = nn.GRU(in_sz, out_sz, hid, batch_first=True, bidirectional=False)
        self.ATTN = Temp_AT(out_sz)
        self.FC = nn.Sequential(
            nn.Linear(out_sz, out_sz//16),
            nn.ReLU(),
            nn.Linear(out_sz//16, 60),
            nn.ReLU()
            )
    def forward(self, x):
        bz = x.size(0)
        x_GRU, hn = self.GRU(x)
        print(x_GRU.size())        
        x_GRU_AT, AT_map = self.ATTN(x_GRU)
        y = self.FC(x_GRU_AT)
        return y.view(bz, -1), AT_map


#%% Test
IN = torch.rand(32,60,68)
F = model_01(68, 68, 3)
Y, AT = F(IN)
print("Pred >>", Y.size())
print("AT >>", AT.size())
