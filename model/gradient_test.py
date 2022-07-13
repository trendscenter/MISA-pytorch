from sklearn.metrics import log_loss
import torch
import numpy as np
import MISAK
# from MISA-loss.py import loss()

grad_desc = torch.optim.SGD(MISAK.output, lr = 0.1)
grad_desc.zero_grad()
# insert loss function callback here
grad_desc.step()


