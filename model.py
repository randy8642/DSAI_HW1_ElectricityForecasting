import torch
import torch.nn as nn


class ElectricityForecastingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weekEmb = nn.Embedding(53, 100)
        self.dayEmb = nn.Embedding(7, 100)

        self.layers = nn.Sequential(
            nn.Linear(201, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.Linear(50, 1),
        )

    def forward(self, year, weekcount, daycount):

        year = year.reshape(-1, 1)
        weekcount = self.weekEmb(weekcount-1)
        daycount = self.dayEmb(daycount-1)
        
        x = torch.cat((year, weekcount, daycount), dim=1)
        x = self.layers(x)
        return x
