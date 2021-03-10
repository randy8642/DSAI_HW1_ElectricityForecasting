import torch.nn as nn
import torch

#%% M
class model_01(nn.Module):
    def __init__(self, in_sz, out_sz, hid):
        super(model_01, self).__init__()
        self.GRU = nn.GRU(in_sz, out_sz, hid, batch_first=True, bidirectional=False)
        seq = 60*out_sz
        self.FC = nn.Sequential(
            nn.Linear(seq, seq//16),
            nn.ReLU(),
            nn.Linear(seq//16, 60)
            )
    def forward(self, x):
        bz = x.size(0)
        x_GRU, hn = self.GRU(x)
        y = self.FC(x_GRU.reshape(bz, -1))
        return y.view(bz, -1)


#%% Test
if __name__ == '__main__':
    IN = torch.rand(32,60,4)
    F = model_01(4, 20, 3)
    Y = F(IN)
    print("Pred >>", Y.size())
