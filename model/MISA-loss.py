import torch
import torch.nn as nn
import numpy as np
import math
from scipy.special import gamma
import MISAK
from subprocess import call
from MISAK import __init__

def loss(self):
	JE = 0
	JF = 0
	JC = 0
	JD = 0
	fc = 0
	MISAK.nes
	for kk in list(torch.where(MISAK.nes)[0].numpy()): #[i for i, l in enumerate(MISAK.nes) if l == True]:
		y_sub = torch.zeros(MISAK.output[0].shape[0], MISAK.d[kk].int()) # self.y_sub = [ for i in range(self.nes.T)], [tot = 0 for i in range(self.nes.T)]
		tot = 0
		for mm in range(MISAK.index.stop)[MISAK.index]:
			ix = slice(tot, tot + MISAK.d_k[mm][kk].int()) # self.ix = [tot + self.d_k[:m][:i] if self.ix is None, 'y_sub(ix,:) = O.Y{mm}(logical(O.S{mm}(kk,:)),:)' else for m in range(self.index)]
			if ix.start < ix.stop:
				y_sub[:,ix] = MISAK.output[mm][:,MISAK.subspace[mm][kk,:]==1] # Unknown for use of boolean in MATLAB code; Purpose of ix?
			tot = tot + MISAK.d_k[mm][kk].int()
		yyT = y_sub.T @ y_sub # torch.Tensor.detach(y_sub.T @ y_sub).numpy()
		# import pdb; pdb.set_trace()
		g_k = torch.pow(torch.diag(yyT),-.5)
		g2_k = torch.pow(torch.diag(yyT), -1)
		g_kInv = torch.pow(torch.diag(yyT),.5)
		ybar_sub = g_kInv * y_sub
		yyTInv = torch.linalg.inv(yyT)# Old line: yyTInv =  yyT / torch.diagonal(torch.eye(MISAK.d[kk].int()), offset = 0) # torch.tensor(np.linalg.lstsq(yyT, torch.eye(MISAK.d[kk].int())))
		A = ybar_sub @ yyTInv # ybar_sub =? torch.zeros_like(yyTInv)
		z_k = torch.sum(ybar_sub * A, axis = 1)
		z_k_beta = torch.pow(z_k, MISAK.beta[kk])
		JE = JE + MISAK.lam[kk] * torch.mean(z_k_beta)
		if MISAK.eta[kk] != 1:
			JF = JF + (1-MISAK.eta[kk]) * torch.mean(torch.log(z_k)) # "91 -" relevance?
		### JC = JC + torch.sum(torch.log(torch.tensor([np.linalg.eig(np.isnan(g_k * yyT * g_k)))[i] for i in range(len(np.isnan(torch.Tensor.detach(g_k).numpy() * (torch.Tensor.detach(yyT).numpy() * torch.Tensor.detach(g_k).numpy()))) - 2)])))
		JC = JC + torch.sum(torch.log(torch.linalg.eig(g_k.T * (yyT * g_k.T))[0])) # [np.linalg.eig(np.isnan(g_k * (yyT * g_k))[i] for i in range(len(np.isnan(g_k* (yyT * g_k)) - 2)))])) # RuntimeError: input should not contain infs or NaNs
		# insert gradient descent here
		JC = JC / 2
		fc = 0.5 * math.log(math.pi) + torch.sum(gamma(MISAK.d)) - torch.sum(torch.log(gamma(0.5 * MISAK.d))) - torch.sum(MISAK.nu * torch.log(MISAK.lam)) - torch.sum(torch.log(MISAK.beta))
		# insert gradient descent here
	J = JE + JF + JC + JD + fc
	return J

model = MISAK.model # (weights = MISAK.weights, index = MISAK.index, subspace = MISAK.subspace, beta = MISAK.beta, eta = MISAK.eta, lam = MISAK.lam, input_dim = MISAK.input_dim, output_dim = MISAK.output_dim)

val = loss(MISAK.output)
print(val)