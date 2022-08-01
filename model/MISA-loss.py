from operator import index
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
	for kk in list(torch.where(MISAK.nes)[0].numpy()):
		y_sub = torch.zeros(MISAK.output[0].shape[0], MISAK.d[kk].int())
		tot = 0
		for mm in range(MISAK.index.stop)[MISAK.index]:
			ix = slice(tot, tot + MISAK.d_k[mm][kk].int())
			if ix.start < ix.stop:
				y_sub[:,ix] = MISAK.output[mm][:,MISAK.subspace[mm][kk,:]==1]
			tot = tot + MISAK.d_k[mm][kk].int()
		yyT = y_sub.T @ y_sub
		g_k = torch.pow(torch.diag(yyT),-.5)
		g2_k = torch.pow(torch.diag(yyT), -1)
		g_kInv = torch.pow(torch.diag(yyT),.5)
		ybar_sub = g_kInv * y_sub
		yyTInv = torch.linalg.inv(yyT)
		A = ybar_sub @ yyTInv
		z_k = torch.sum(ybar_sub * A, axis = 1)
		z_k_beta = torch.pow(z_k, MISAK.beta[kk])
		JE = JE + MISAK.lam[kk] * torch.mean(z_k_beta)
		if MISAK.eta[kk] != 1:
			JF = JF + (1-MISAK.eta[kk]) * torch.mean(torch.log(z_k))
		JC = JC + torch.sum(torch.log(torch.linalg.eig(g_k[:,None] * (yyT * g_k[None,:]))[0]))
		JC = JC / 2
		fc = 0.5 * torch.log(torch.tensor(torch.pi)) + torch.sum(torch.lgamma(MISAK.d)) - torch.sum(torch.lgamma(0.5 * MISAK.d)) - torch.sum(MISAK.nu * torch.log(MISAK.lam)) - torch.sum(torch.log(MISAK.beta))
		for mm in range(MISAK.index.stop)[MISAK.index]:
			[rr, cc] = torch.tensor(MISAK.net.state_dict()['parameters']).size()
			if rr == cc:
				JD = JD - torch.log(torch.abs(torch.det(MISAK.net[mm])))
			else:
				D = torch.linalg.eig(MISAK.net[mm] * MISAK.net.T[mm])[0]
				JD = JD - torch.sum(torch.log(torch.abs(torch.sqrt(D))))
				del D
	J = JE + JF + JC + JD + fc
	return J

if __name__ == '__main__':
	model = MISAK.model
	val = loss(MISAK.output)
	print(val)