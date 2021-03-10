import torch
import torch.nn as nn


class ElectricityForecastingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(3,10),
            nn.BatchNorm1d(10),
            nn.Linear(10,2),
        )

    def forward(self, x):
        
        
        x = self.layers(x)
        return x
