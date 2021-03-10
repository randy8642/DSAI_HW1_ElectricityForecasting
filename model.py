import torch
import torch.nn as nn


class ElectricityForecastingModel(nn.Module):
    def __init__(self):
        super().__init__()


        self.monthEmb = nn.Embedding(32,100)
        self.dayEmb = nn.Embedding(32,100)

        self.layers = nn.Sequential(
            nn.Linear(100,100),
            nn.BatchNorm1d(10),
            nn.Linear(100,50),
            nn.BatchNorm1d(50),
            nn.Linear(50,2),
        )

    def forward(self, x):
        year, month, day = x[:,0],x[:,1],x[:,2]
        
        x = self.layers(x)
        return x
