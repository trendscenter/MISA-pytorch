import copy
import torch
import torch.nn as nn
import random
import numpy as np
import scipy.io as sio
from metrics.misi import MISI

class MISA(nn.Module):
    def __init__(self, weights=list(), index=None, subspace=list(), beta=0.5, eta=1, lam=1, input_dim=list(), output_dim=list(), bias=False, seed=0, device='cpu'):
        # Read identity matrix from columns to rows, ie source = columns and subspace = rows
        super(MISA, self).__init__()
        self.seed()
        # Setting arguments/filters
        assert torch.all(torch.gt(eta, (2-torch.sum(torch.cat(subspace[index], axis=1), axis=1))/2)).item(), "All eta parameters should be lagerer than (2-d)/2."
        assert torch.all(torch.gt(beta, torch.zeros_like(beta))).item(), "All beta parameters should be positive"  # Boolean operations in torch
        assert torch.all(torch.gt(lam, torch.zeros_like(lam))).item(), "All lambda parameters should be positive"
        nu = (2*eta + torch.sum(torch.cat(subspace[index], axis=1), dim=1) - 2)/(2*beta)
        assert torch.all(torch.gt(nu, torch.zeros_like(nu))).item(), "All nu parameter derived from eta and d should be positive."
        # Defining variables
        self.device = device
        self.index = index  # M (slice object)
        self.subspace = subspace  # S
        self.beta = beta  # beta
        self.eta = eta  # eta
        self.lam = lam  # lambda
        self.nu = nu
        if weights != []:
            self.input_dim = [weights[i].shape[1] for i in range(len(weights))]
            self.output_dim = [weights[i].shape[0] for i in range(len(weights))]
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim
        self.net = nn.ModuleList([nn.Linear(self.input_dim[i], self.output_dim[i], bias = bias) if i in range(self.index.stop)[self.index] else None for i in range(self.index.stop)])  # Optional argument/parameter (w0)
        self.output = list()
        self.num_observations = None
        self.d = torch.sum(torch.cat(self.subspace[self.index], axis=1), axis=1)
        self.nes = torch.ne(self.d, torch.zeros_like(self.d, device=device))
        self.a = (torch.pow(self.lam, (-1/self.beta))) * torch.lgamma(self.nu + 1 / self.beta) / (self.d * torch.lgamma(self.nu))
        self.d_k = [(torch.sum(self.subspace[i].int(), axis=1)).int() for i in range(len(self.subspace))]
        self.K = self.nes.sum()
        if weights != []:
            for mm in range(self.index.stop)[self.index]:
                with torch.no_grad():
                    self.net[mm].weight.copy_(torch.from_numpy(weights[mm])) 

    def seed(self, seed = None, seed_torch = True):
        if seed is None:
            self.seed = np.random.choice(2 ** 32)
            random.seed(self.seed)
            np.random.seed(self.seed)
        if seed_torch:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        torch.random.seed(worker_seed)
        random.seed(worker_seed)

    def forward(self, x):
        self.output = [l(x[i]) if isinstance(l, nn.Linear) else None for i, l in enumerate(self.net)]

    def loss(self):
        JE = 0
        JF = 0
        JC = 0
        JD = 0
        fc = 0
        self.nes
        for kk in list(torch.where(self.nes)[0]):
            y_sub = torch.zeros(self.output[0].shape[0], self.d[kk].int(), device=self.device)
            tot = 0
            for mm in range(self.index.stop)[self.index]:
                ix = slice(tot, tot + self.d_k[mm][kk].int())
                # print("ysub", y_sub.shape, " ix ", ix, " self output shape ", self.output[mm].shape, ' subspace shape ', self.subspace[mm][kk, :])
                if ix.start < ix.stop:
                    y_sub[:, ix] = self.output[mm][:,self.subspace[mm][kk, :] == 1]
                tot = tot + self.d_k[mm][kk].int()
            yyT = y_sub.T @ y_sub
            g_k = torch.pow(torch.diag(yyT), -.5)
            g2_k = torch.pow(torch.diag(yyT), -1)
            g_kInv = torch.pow(torch.diag(yyT), .5)
            ybar_sub = g_kInv * y_sub
            yyTInv = torch.linalg.inv(yyT)
            A = ybar_sub @ yyTInv
            z_k = torch.sum(ybar_sub * A, axis=1)
            z_k_beta = torch.pow(z_k, self.beta[kk])
            JE = JE + self.lam[kk] * torch.mean(z_k_beta)
            if self.eta[kk] != 1:
                JF = JF + (1-self.eta[kk]) * torch.mean(torch.log(z_k))
            JC = JC + torch.sum(torch.log(torch.linalg.eigvalsh(g_k[:, None] * (yyT * g_k[None, :]))))
        JC = JC / 2
        fc = 0.5 * torch.log(torch.tensor(torch.pi)) * torch.sum(self.d) + torch.sum(torch.lgamma(self.nu)) - torch.sum(torch.lgamma(0.5 * self.d)) - torch.sum(self.nu * torch.log(self.lam)) - torch.sum(torch.log(self.beta))
        for mm in range(self.index.stop)[self.index]:
            cc,rr = self.net[mm].weight.size()
            if rr == cc:
                JD = JD - torch.linalg.slogdet(self.net[mm].weight)[1]
            else:
                D = torch.linalg.eigvalsh(self.net[mm].weight @ self.net[mm].weight.T)
                JD = JD - torch.sum(torch.log(torch.abs(torch.sqrt(D))))
        J = JE + JF + JC + JD + fc
        return J
    def train_me(self, train_data, n_iter, learning_rate, A, beta1, beta2, batch_size, weights, seed, patience, fused, foreach):
        optim = torch.optim.Adam(self.parameters(), lr = learning_rate, betas=(beta1, beta2), fused=fused, foreach=foreach)
        training_loss = []
        batch_loss = []
        training_MISI = []        
        trigger_times = 0
        nn_weight_threshold = 1e-4
        loss_threshold = 0.1
        # patience = 2
        
        for epochs in range(n_iter):
            for i, data in enumerate(train_data, 0):
                optim.zero_grad()
                self.forward(data)
                loss = self.loss()
                loss.backward()
                optim.step()
                batch_loss.append(loss.detach())
            training_loss.append(batch_loss)
            
            if A is not None:
                training_MISI.append(MISI([nn.weight.detach().cpu().numpy() for nn in self.net],A,[ss.detach().cpu().numpy() for ss in self.subspace])[0])
                print('epoch: {} \tloss: {} \tMISI: {}'.format(epochs+1, loss.detach().cpu().numpy(), training_MISI[-1]))
            else:
                print('epoch: {} \tloss: {}'.format(epochs+1, loss.detach().cpu().numpy()))
            
            # early stop
            if epochs == 0: 
                nn_weight_current = [nn.weight.detach().cpu().numpy() for nn in self.net]
                nn_weight_previous = copy.deepcopy(nn_weight_current)
                loss_current = loss.detach().cpu().numpy()
                loss_previous = loss_current
            else:
                nn_weight_current = [nn.weight.detach().cpu().numpy() for nn in self.net]
                nn_weight_diff = np.max(np.array([np.max(np.abs(nn_weight_previous[i]-c)) for i, c in enumerate(nn_weight_current)]))
                loss_current = loss.detach().cpu().numpy()
                loss_diff = np.abs(loss_current-loss_previous)
                if nn_weight_diff < nn_weight_threshold or loss_diff < loss_threshold:
                    trigger_times += 1
                    print(f'Trigger Times: {trigger_times}')
                    if trigger_times > patience:
                        print(f'Early stopping! \nThe maximum absolute difference of W matrix is less than {nn_weight_threshold} or that of loss is less than {loss_threshold} between the previous and current iteration for {trigger_times} iterations.')
                        epochs_completed = epochs
                        return training_loss, training_MISI, optim, epochs_completed
                else:
                    trigger_times = 0
                nn_weight_previous = copy.deepcopy(nn_weight_current)
                loss_previous = loss_current
                epochs_completed = epochs
        return training_loss, training_MISI, optim, epochs_completed

    def predict(self, test_data):
        test_loss = []
        for i, data in enumerate(test_data, 0):
            self.forward(data)
            loss = self.loss()
            test_loss.append(loss.detach())
        return test_loss

if __name__ == "__main__":
    X_mat = sio.loadmat("simulation_data/X.mat")['X'].squeeze()
    x = [torch.tensor(np.float32(X_mat[i].T)) for i in range(len(X_mat))]
    N = x[0].size(0)
    num_modal = len(x)
    index = slice(0, num_modal)
    W0_mat = sio.loadmat("simulation_data/W0.mat")['W0'].squeeze()
    w = [torch.tensor(np.float32(W0_mat[i])) for i in range(len(W0_mat))]
    K = w[0].size(1) # number of subspaces
    subspace = [torch.eye(K) for _ in range(num_modal)]
    beta = 0.5 * torch.ones(K)
    eta = torch.ones(K)
    lam = torch.ones(K)
    model = MISA(weights=w, index=index, subspace=subspace, beta=beta, eta=eta, lam=lam, input_dim=[], output_dim=[])
    model.cuda()
    model.forward(x)
    print(model.output)
    loss = model.loss()
    print(loss)