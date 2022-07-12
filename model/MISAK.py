from tkinter import N
import torch
import torch.nn as nn
import numpy as np
import math
from scipy.special import gamma
import random

class MISA(nn.Module):
	def __init__(self, weights = list(), index = None, subspace = list(), beta = 0.5, eta = 1, lam = 1, input_dim = list(), output_dim = list()): # Unknown default parameters, "weights = None" for optional weight modification
		# Read identity matrix from columns to rows, ie source = columns and subspace = rows
		super(MISA, self).__init__()
		# Setting arguments/filters
		assert torch.all(torch.gt(beta, torch.zeros_like(beta))).item(), "All beta parameters should be positive" # Boolean operations in torch
		assert torch.all(torch.gt(lam, torch.zeros_like(lam))).item(), "All lambda parameters should be positive"
		assert torch.all(torch.gt(eta, (2-torch.sum(torch.cat(subspace[index], axis = 1), axis = 1))/2)).item(), "All eta parameters should be lagerer than (2-d)/2."
		nu = (2*eta + torch.sum(torch.cat(subspace[index], axis = 1), dim = 1) - 2)/(2*beta)
		assert torch.all(torch.gt(nu, torch.zeros_like(nu))).item(), "All nu parameter derived from eta and d should be positive."
		# Defining variables
		self.index = index # M (slice object)
		self.subspace = subspace # S
		self.beta = beta # beta
		self.eta = eta # eta
		self.lam = lam # lambda
		self.nu = nu
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.net = nn.ModuleList([nn.Linear(self.input_dim[i], self.output_dim[i]) if i in range(self.index.stop)[self.index] else None for i in range(self.index.stop)])  # Optional argument/parameter (w0), why the ".stop" add on? What are the dimensions of this layer exactly?
		self.output = list()
		self.num_observations = None
		self.d = torch.sum(torch.cat(self.subspace[self.index], axis = 1), axis = 1)
		self.nes = torch.ne(self.d, torch.zeros_like(self.d))
		self.a = (torch.pow(self.lam,(-1/self.beta))) * gamma(self.nu + 1 / self.beta) / (self.d * gamma(self.nu)) # "NameError: name 'self' is not defined" despite passing through debugger
		self.d_k = [(torch.sum(self.subspace[i].int(), axis = 1)).int() for i in range(len(self.subspace))]
		self.K = self.nes.sum() # Unsure if consistent to summing up "True" elements or if added "False" and was coincidential
	def seed():
		random.seed()
	def forward(self, x):
		self.output = [l(x[i]) if isinstance(l, nn.Linear) else None for i, l in enumerate(self.net)]
	def loss(self):
		loss = torch.nn.CrossEntropyLoss()
		output = loss("Insert input here", self.y)
		output.backward()
	def train(self, x):
		# Placeholder for later development
		print("Test")
	def predict(self, x):
		# Placeholder for later development
		print("Test")

beta = 0.5 * torch.ones(5)
eta = torch.ones(5)
num_modal = 3
index = slice(0,num_modal)# [i for i in range(num_modal)]
subspace = [torch.eye(5) for _ in range(num_modal)] #Alternatives to torch.eye()? Columns in torch.eye needs to match output_dim list
lam = torch.ones(5)
input_dim = [6,6,6]
output_dim = [5,5,5]
model = MISA(weights = list(), index = index, subspace = subspace, beta = beta, eta = eta, lam = lam, input_dim = input_dim, output_dim = output_dim)
N = 1000
x = [torch.rand(N,d) if i in range(index.stop)[index] else None for i, d in enumerate(input_dim)]
model.forward(x)
print(model.output)
nes = model.nes
d = torch.sum(torch.cat(model.subspace[model.index], axis = 1), axis = 1)
num_observations = None
d_k = [torch.sum(model.subspace[i], axis = 1) for i in range(len(model.subspace))]
output = model.output
weights = list()
nu = model.nu
net = model.net