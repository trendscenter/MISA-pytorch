from modulefinder import Module
from tkinter import Y
import torch
import torch.nn as nn
import numpy
# from base import BaseModel

class LinearModel(nn.Module):
    def __init__(self, input = 1, hidden = 2, output = 1, number_models = 1, bias = True):
        super(LinearModel, self).__init__()
        self.input = input
        self.hidden = hidden
        self.output = output
        self.number_models = number_models
        # self.linears = nn.ModuleList([nn.Linear(self.input, self.hidden, bias) for i in range(self.number_models)])
        self.epoch = 0
        self.b = nn.ModuleList()
        for _ in range(self.number_models):
            self.b.append(nn.Sequential(
                nn.Linear(self.input, self.hidden, bias),
                nn.ReLU(),
                nn.Linear(self.hidden, self.output, bias)
            ))
    def forward(self, x):
        self.y = [l(x[i]) for i, l in enumerate(self.b)]
        
    def forward2(self, x):
        self.y2 = [l(x[m]) for m, l in enumerate(self.b)]
        # self.y2 = [l(torch.tensor(x)) for _, l in enumerate(self.linears)]
        # return x
        

# x = [torch.randn(6,1) for k in range(5)]
x = [torch.ones(1,2) for m in range(2)]
model = LinearModel(input = 2, hidden = 4, output = 3, number_models = 2)
model.forward(x)
print("Output forward 1: " + str(model.y))
# model.forward2(x)
# print("Output model.y: " + str(model.y2))
print("X matrix: " + str(x))