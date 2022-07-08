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
	for kk in [i for i, l in enumerate(MISAK.nes) if l == True]:
		y_sub = torch.zeros(MISAK.output[0].shape[0], MISAK.d[kk].int()) # self.y_sub = [ for i in range(self.nes.T)], [tot = 0 for i in range(self.nes.T)]
		tot = 0
		for mm in range(MISAK.index.stop)[MISAK.index]:
			ix = slice(tot + 0, tot + MISAK.d_k[mm][kk].item()) # self.ix = [tot + self.d_k[:m][:i] if self.ix is None, 'y_sub(ix,:) = O.Y{mm}(logical(O.S{mm}(kk,:)),:)' else for m in range(self.index)]
			if ix is not None:
				y_sub[:,int(ix.stop)-1] = torch.squeeze(MISAK.output[mm][:,MISAK.subspace[mm][kk,:]==1])
			tot = tot + MISAK.d_k[mm][kk]
	yyT = torch.Tensor.detach(y_sub.T @ y_sub).numpy()
	g_k = pow(np.diag(yyT),-.5)
	g2_k = pow(np.diag(yyT), -1)
	g_kInv = pow(np.diag(yyT),.5)
	ybar_sub = torch.tensor(g_kInv).reshape(3,1) * y_sub.T
	yyTInv = np.linalg.inv(yyT) * np.eye(MISAK.d[kk].int())
	A = torch.tensor(yyTInv @ ybar_sub.cpu().detach().numpy())
	z_k = torch.sum(ybar_sub * A, 0)
	z_k_beta = pow(z_k, MISAK.beta[kk])
	JE = JE + MISAK.lam[kk] * torch.mean(z_k_beta)
	if MISAK.eta[kk] != 1:
		JF = JF + (1-MISAK.eta[kk]) * torch.mean(torch.log(z_k))
	# import pdb; pdb.set_trace()
	# why negative eigenvalues?
	JC = JC + np.sum(np.log(np.linalg.eig(g_k * (yyT * g_k))[0]))
	# insert gradient descent here
	JC = JC / 2
	fc = 0.5 * math.log(math.pi) + torch.sum(gamma(MISAK.d)) - torch.sum(torch.log(gamma(0.5 * MISAK.d))) - torch.sum(MISAK.nu * torch.log(MISAK.lam)) - torch.sum(torch.log(MISAK.beta))
	# insert gradient descent here
	J = JE + JF + JC + JD + fc
	return J

model = MISAK.model # (weights = MISAK.weights, index = MISAK.index, subspace = MISAK.subspace, beta = MISAK.beta, eta = MISAK.eta, lam = MISAK.lam, input_dim = MISAK.input_dim, output_dim = MISAK.output_dim)

val = loss(MISAK.output)
print(val)