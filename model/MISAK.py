import copy
import torch
import torch.nn as nn
import random
import numpy as np
import scipy.io as sio
from metrics.misi import MISI

class MISA(nn.Module):
    def __init__(self, weights=list(), index=None, subspace=list(), beta=0.5, eta=1, lam=1, input_dim=list(), output_dim=list(), bias=False, seed=0, device='cpu', model=None, latent_dim=None):
        # Read identity matrix from columns to rows, ie source = columns and subspace = rows
        super(MISA, self).__init__()
        self.seed = seed
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
        self.input_model = model
        self.n_modality = index.stop
        self.latent_dim = latent_dim
        if self.input_model:
            # self.input_dim = self.input_model.output_dim # TODO revisit: How to obtain iVAE output_dim
            if weights != []:
                self.input_dim = [weights[i].shape[1] for i in range(len(weights))]
                self.output_dim = [weights[i].shape[0] for i in range(len(weights))]
            else:
                self.input_dim = input_dim
                self.output_dim = output_dim
        elif weights != []:
            self.input_dim = [weights[i].shape[1] for i in range(len(weights))]
            self.output_dim = [weights[i].shape[0] for i in range(len(weights))]
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim
        if model is None:
            self.net = nn.ModuleList([nn.Linear(self.input_dim[i], self.output_dim[i], bias = bias) if i in range(self.n_modality)[self.index] else None for i in range(self.n_modality)])  # Optional argument/parameter (w0)
        else:
            # TODO revisit: Combine modality-specific model weights
            self.net = None
            # self.net = nn.ModuleList([self.input_model[m].g.fc[-1] for m in range(self.n_modality)])
        self.output = list()
        self.num_observations = None
        self.d = torch.sum(torch.cat(self.subspace[self.index], axis=1), axis=1)
        self.nes = torch.ne(self.d, torch.zeros_like(self.d, device=device))
        self.a = (torch.pow(self.lam, (-1/self.beta))) * torch.lgamma(self.nu + 1 / self.beta) / (self.d * torch.lgamma(self.nu))
        self.d_k = [(torch.sum(self.subspace[i].int(), axis=1)).int() for i in range(len(self.subspace))]
        self.K = self.nes.sum()
        if weights == "mgpca":
            pass
        elif weights != []:
            for m in range(self.n_modality)[self.index]:
                with torch.no_grad():
                    self.net[m].weight.copy_(torch.from_numpy(weights[m])) 

    # def seed(self, seed = None, seed_torch = True):
    #     if seed is None:
    #         self.seed = np.random.choice(2 ** 32)
    #         random.seed(self.seed)
    #         np.random.seed(self.seed)
    #     if seed_torch:
    #         print(self.seed)
    #         torch.manual_seed(self.seed)
    #         torch.cuda.manual_seed_all(self.seed)
    #         torch.cuda.manual_seed(self.seed)
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        torch.random.seed(worker_seed)
        random.seed(worker_seed)

    def forward(self, x):
        if self.input_model is None:
            # IVA
            self.output = [l(x[i]) if isinstance(l, nn.Linear) else None for i, l in enumerate(self.net)]
        else:
            # iVAE
            z = []
            for m in range(len(self.input_model)):
                encoder_params = self.input_model[m].encoder_params(x[0][:,:,m], x[1]) #x[0]: input data, x[1]: auxiliary information
                z.append(self.input_model[m].encoder_dist.sample(*encoder_params))
            self.output = z

    def loss(self, x=None, approximate_jacobian=True):
        JE = 0
        JF = 0
        JC = 0
        JD = 0
        fc = 0
        self.nes
        for kk in list(torch.where(self.nes)[0]):
            y_sub = torch.zeros(self.output[0].shape[0], self.d[kk].int(), device=self.device)
            tot = 0
            for m in range(self.n_modality):
                ix = slice(tot, tot + self.d_k[m][kk].int())
                if ix.start < ix.stop:
                    y_sub[:, ix] = self.output[m][:,self.subspace[m][kk, :] == 1]
                tot = tot + self.d_k[m][kk].int()
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
        
        def compute_JD(jacobian):
            jacobian = jacobian.squeeze()
            cc,rr = jacobian.size()
            if rr == cc:
                # input: sample x n_source x (n_source + n_segment)
                jd = torch.linalg.slogdet(jacobian)[1]
            else:
                # rectangular matrix
                jacobian_sq = jacobian @ jacobian.T
                try:
                    D = torch.linalg.eigvalsh(jacobian_sq)
                except:
                    # add random noise to avoid ill-conditioned matrix
                    noise = torch.normal(0, 0.1, size = jacobian_sq.size())
                    D = torch.linalg.eigvalsh(jacobian_sq + noise)
                    jd = torch.sum(torch.log(torch.abs(torch.sqrt(D))))
            return jd

        for m in range(self.n_modality):
            if self.net is not None:
                net_fc = self.net[m]
                jacobian = [net_fc.weight]
            else:
                # Jacobian of encoder weights
                jacobian = [ torch.autograd.functional.jacobian( self.input_model[m].g, torch.unsqueeze(x[0][i,:,m], 0) ) for i in range(x[0].size()[0])] #x[0]: input data, x[1]: auxiliary information
                if approximate_jacobian:
                    jacobian = [torch.mean(torch.stack(jacobian), dim = 0)]
            
            jd = []
            for j in jacobian:
                jd.append(compute_JD(j))
            JD = JD - torch.mean(torch.stack(jd))
            
            # cc,rr = jacobian.size()
            # if rr == cc:
            #     # input: sample x n_source x (n_source + n_segment)
            #     JD = JD - torch.linalg.slogdet(jacobian)[1]
            # else:
            #     # rectangular matrix
            #     jacobian_sq = jacobian @ jacobian.T
            #     try:
            #         D = torch.linalg.eigvalsh(jacobian_sq)
            #     except:
            #         # add random noise to avoid ill-conditioned matrix
            #         noise = torch.normal(0, 0.1, size = jacobian_sq.size())
            #         D = torch.linalg.eigvalsh(jacobian_sq + noise)
            #     JD = JD - torch.sum(torch.log(torch.abs(torch.sqrt(D))))
        
        J = JE + JF + JC + JD + fc
        return J

    def train_me(self, train_data, n_iter, learning_rate, A=None):
        if self.input_model is None:
            params = self.parameters()
        else:
            params = []
            for i in range(self.n_modality):
                params += list(self.input_model[i].parameters()) # remove rows corresponding to u, also look into requires_grad
        optimizer = torch.optim.Adam(params, lr = learning_rate)
        # TODO implement a scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
        training_loss = []
        batch_loss = []
        training_MISI = []        
        trigger_times = 0
        nn_weight_threshold = 1e-12
        loss_threshold = 1e-4
        patience = 2
        
        for it in range(n_iter):
            for i, data in enumerate(train_data, 0):
                optimizer.zero_grad()
                self.forward(data)
                loss = self.loss(x=data)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.detach())
            training_loss.append(batch_loss)
            # scheduler.step(np.mean(np.array(batch_loss)))
            
            if A is not None:
                training_MISI.append(MISI([nn.weight.detach().cpu().numpy() for nn in self.net],A,[ss.detach().cpu().numpy() for ss in self.subspace])[0])
                print('MISA epoch: {} \tloss: {} \tMISI: {}'.format(it+1, loss.detach().cpu().numpy(), training_MISI[-1]))
            else:
                print('MISA epoch: {} \tloss: {}'.format(it+1, loss.detach().cpu().numpy()))
            
            # early stop
            # if it == 0: 
            #     nn_weight_current = [nn.weight.detach().cpu().numpy() for nn in self.net]
            #     nn_weight_previous = copy.deepcopy(nn_weight_current)
            #     loss_current = loss.detach().cpu().numpy()
            #     loss_previous = loss_current
            # else:
            #     nn_weight_current = [nn.weight.detach().cpu().numpy() for nn in self.net]
            #     nn_weight_diff = np.max(np.array([np.max(np.abs(nn_weight_previous[i]-c)) for i, c in enumerate(nn_weight_current)]))
            #     loss_current = loss.detach().cpu().numpy()
            #     loss_diff = np.abs(loss_current-loss_previous)
            #     if nn_weight_diff < nn_weight_threshold or loss_diff < loss_threshold:
            #         trigger_times += 1
            #         print(f'Trigger Times: {trigger_times}; Maximum W Difference: {nn_weight_diff}; Loss Difference: {loss_diff}')
            #         if trigger_times > patience:
            #             print(f'Early stopping! \nThe maximum absolute difference of W matrix is less than {nn_weight_threshold} or that of loss is less than {loss_threshold} between the previous and current iteration for {trigger_times} iterations.')
            #             return training_loss, training_MISI, optimizer
            #     else:
            #         trigger_times = 0
            #     nn_weight_previous = copy.deepcopy(nn_weight_current)
            #     loss_previous = loss_current

        return training_loss, training_MISI, optimizer

    def predict(self, test_data):
        test_loss = []
        for i, data in enumerate(test_data, 0):
            self.forward(data)
            loss = self.loss(x=data)
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
