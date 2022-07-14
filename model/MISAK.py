import torch
import torch.nn as nn
import random

class MISA(nn.Module):
    # Unknown default parameters, "weights = None" for optional weight modification
    def __init__(self, weights=list(), index=None, subspace=list(), beta=0.5, eta=1, lam=1, input_dim=list(), output_dim=list()):
        # Read identity matrix from columns to rows, ie source = columns and subspace = rows
        super(MISA, self).__init__()
        # Setting arguments/filters
        assert torch.all(torch.gt(beta, torch.zeros_like(beta))).item(), "All beta parameters should be positive"  # Boolean operations in torch
        assert torch.all(torch.gt(lam, torch.zeros_like(lam))).item(), "All lambda parameters should be positive"
        assert torch.all(torch.gt(eta, (2-torch.sum(torch.cat(subspace[index], axis=1), axis=1))/2)).item(), "All eta parameters should be lagerer than (2-d)/2."
        nu = (2*eta + torch.sum(torch.cat(subspace[index], axis=1), dim=1) - 2)/(2*beta)
        assert torch.all(torch.gt(nu, torch.zeros_like(nu))).item(), "All nu parameter derived from eta and d should be positive."
        # Defining variables
        self.index = index  # M (slice object)
        self.subspace = subspace  # S
        self.beta = beta  # beta
        self.eta = eta  # eta
        self.lam = lam  # lambda
        self.nu = nu
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.ModuleList([nn.Linear(self.input_dim[i], self.output_dim[i]) if i in range(self.index.stop)[self.index] else None for i in range(self.index.stop)])  # Optional argument/parameter (w0), why the ".stop" add on? What are the dimensions of this layer exactly?
        self.output = list()
        self.num_observations = None
        self.d = torch.sum(torch.cat(self.subspace[self.index], axis=1), axis=1)
        self.nes = torch.ne(self.d, torch.zeros_like(self.d))
        # "NameError: name 'self' is not defined" despite passing through debugger
        self.a = (torch.pow(self.lam, (-1/self.beta))) * torch.lgamma(self.nu + 1 / self.beta) / (self.d * torch.lgamma(self.nu))
        self.d_k = [(torch.sum(self.subspace[i].int(), axis=1)).int() for i in range(len(self.subspace))]
        # Unsure if consistent to summing up "True" elements or if added "False" and was coincidential
        self.K = self.nes.sum()

    def seed(seed = None, seed_torch = True):
        if seed is None:
            seed = torch.random.choice(2 ** 32)
            random.seed(seed)
            torch.random.seed(seed)
        if seed_torch:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
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
        # [i for i, l in enumerate(MISAK.nes) if l == True]:
        for kk in list(torch.where(self.nes)[0].numpy()):
            # self.y_sub = [ for i in range(self.nes.T)], [tot = 0 for i in range(self.nes.T)]
            y_sub = torch.zeros(self.output[0].shape[0], self.d[kk].int())
            tot = 0
            for mm in range(self.index.stop)[self.index]:
                # self.ix = [tot + self.d_k[:m][:i] if self.ix is None, 'y_sub(ix,:) = O.Y{mm}(logical(O.S{mm}(kk,:)),:)' else for m in range(self.index)]
                ix = slice(tot, tot + self.d_k[mm][kk].int())
                if ix.start < ix.stop:
                    # Unknown for use of boolean in MATLAB code; Purpose of ix?
                    y_sub[:, ix] = self.output[mm][:,self.subspace[mm][kk, :] == 1]
                tot = tot + self.d_k[mm][kk].int()
            # torch.Tensor.detach(y_sub.T @ y_sub).numpy()
            yyT = y_sub.T @ y_sub
            # import pdb; pdb.set_trace()
            g_k = torch.pow(torch.diag(yyT), -.5)
            g2_k = torch.pow(torch.diag(yyT), -1)
            g_kInv = torch.pow(torch.diag(yyT), .5)
            ybar_sub = g_kInv * y_sub
            # Old line: yyTInv =  yyT / torch.diagonal(torch.eye(MISAK.d[kk].int()), offset = 0) # torch.tensor(np.linalg.lstsq(yyT, torch.eye(MISAK.d[kk].int())))
            yyTInv = torch.linalg.inv(yyT)
            A = ybar_sub @ yyTInv  # ybar_sub =? torch.zeros_like(yyTInv)
            z_k = torch.sum(ybar_sub * A, axis=1)
            z_k_beta = torch.pow(z_k, self.beta[kk])
            JE = JE + self.lam[kk] * torch.mean(z_k_beta)
            if self.eta[kk] != 1:
                # "91 -" relevance?
                JF = JF + (1-self.eta[kk]) * torch.mean(torch.log(z_k))
            # JC = JC + torch.sum(torch.log(torch.tensor([np.linalg.eig(np.isnan(g_k * yyT * g_k)))[i] for i in range(len(np.isnan(torch.Tensor.detach(g_k).numpy() * (torch.Tensor.detach(yyT).numpy() * torch.Tensor.detach(g_k).numpy()))) - 2)])))
            # [np.linalg.eig(np.isnan(g_k * (yyT * g_k))[i] for i in range(len(np.isnan(g_k* (yyT * g_k)) - 2)))])) # RuntimeError: input should not contain infs or NaNs
            JC = JC + torch.sum(torch.log(torch.linalg.eig(g_k[:, None] * (yyT * g_k[None, :]))[0]))
            # insert gradient descent here
            JC = JC / 2
            fc = 0.5 * torch.log(torch.tensor(torch.pi)) + torch.sum(torch.lgamma(self.d)) - torch.sum(torch.lgamma(0.5 * self.d)) - torch.sum(self.nu * torch.log(self.lam)) - torch.sum(torch.log(self.beta))
            # insert gradient descent here
            for mm in range(self.index.stop)[self.index]:
                # torch.tensor(MISAK.net).size()
                [rr, cc] = torch.tensor(self.net[mm].weight).size()# state_dict()['parameters']).size()
                # rr and cc are flipped
                if rr == cc:
                    JD = JD - torch.log(torch.abs(torch.det(self.net[mm])))
                else:
                    D = torch.linalg.eig(self.net[mm] * self.net.T[mm])[0]
                    JD = JD - torch.sum(torch.log(torch.abs(torch.sqrt(D))))
                    del D
        J = JE + JF + JC + JD + fc
        return J

    def training(self, train_data, learning_rate):
        optim = torch.optim.Adam(self.parameters, lr=learning_rate)
        training_loss = []
        batch_loss = []
        for epochs in n_iter:
            for i, data in enumerate(train_data, 0):
                optim.zero_grad()
                self.forward(data)
                loss = self.loss()
                loss.backward()
                optim.step()
                batch_loss.append(loss.detach())
            training_loss.append(batch_loss)
            if epochs % 1 == 0:
                print('epoch ', epochs+1, ' loss = ', loss.detach())

    def predict(self, test_data, learning_rate):
        batch_loss = []
        for i, data in enumerate(test_data, 0):
            self.forward(data)
            loss = self.loss()
            batch_loss.append(loss.detach())
        print('epoch ', epochs+1, ' loss = ', loss.detach())
beta = 0.5 * torch.ones(5)
eta = torch.ones(5)
num_modal = 3
index = slice(0, num_modal)  # [i for i in range(num_modal)]
# Alternatives to torch.eye()? Columns in torch.eye needs to match output_dim list
subspace = [torch.eye(5) for _ in range(num_modal)]
lam = torch.ones(5)
input_dim = [6, 6, 6]
output_dim = [5, 5, 5]
model = MISA(weights=list(), index=index, subspace=subspace, beta=beta, eta=eta, lam=lam, input_dim=input_dim, output_dim=output_dim)
N = 1000
x = [torch.rand(N, d) if i in range(index.stop)[index] else None for i, d in enumerate(input_dim)]
model.forward(x)
print(model.output)
loss = model.loss()
print(loss)
# nes = model.nes
# d = torch.sum(torch.cat(model.subspace[model.index], axis=1), axis=1)
# num_observations = None
# d_k = [torch.sum(model.subspace[i], axis=1) for i in range(len(model.subspace))]
# output = model.output
# weights = list()
# nu = model.nu
# net = model.net
# learning_rate = 0.01
# n_iter = 1000
# train_data = DataLoader("Insert parameters here", lr = learning_rate, shuffle = True)
# test = DataLoader("Insert dataset here", lr = learning_rate, shuffle = True)
# x = [test if i in range(index.stop)[index] else None for i, d in enumerate(input_dim)]