# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:16:54 2021

@author: Lab
"""
import torch
import torch.nn as nn

class m01(nn.Module):
    def __init__(self, in_num, out_num, seq):
        super(m01, self).__init__()
        self.GRU = nn.GRU(in_num, out_num, num_layers=2, batch_first=True, bidirectional=True)
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_num*seq*2, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
            )

    def forward(self, x):
        x_GRU, hn = self.GRU(x)
        pred = self.FC(x_GRU)
        return pred
    
class m02(nn.Module):
    def __init__(self, in_num, seq):
        super(m02, self).__init__()
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_num*seq, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
            )

    def forward(self, x):
        pred = self.FC(x)
        return pred    
    
class m03(nn.Module):
    def __init__(self, in_num, seq):
        super(m03, self).__init__()
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_num*seq, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
            )

    def forward(self, x):
        pred = self.FC(x)
        return pred    
    
#%% Test
if __name__ == "__main__":

    IN = torch.randn(1,30,3)
    F = m01(3, 10, 30)
    F2 = m03(3, 30)
    Pred = F(IN)
    Pred2 = F2(IN)
    
    print("Pred >> ", Pred.size())
    print("Pred2 >> ", Pred2.size())