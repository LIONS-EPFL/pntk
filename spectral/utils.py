import pdb

import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from scipy.special import eval_gegenbauer


class pi_model_ntk(nn.Module):
    def __init__(self, opt):
        super(pi_model_ntk, self).__init__()
        self.degree = opt.degree
        for i in range(opt.degree):
            hidden_layer_1 = nn.Linear(opt.d, opt.WIDTH, bias=opt.bias)
            setattr(self, "layer{}".format(i), nn.Sequential(hidden_layer_1,nn.ReLU()))    
        out_layer = nn.Linear(opt.WIDTH, 1, bias=opt.bias)
        setattr(self, "layer{}".format(self.degree), out_layer)

    def forward(self, x):
        y = x+0
        y = getattr(self, "layer{}".format(0))(y)

        for i in range(1,self.degree,1):
            y = y * getattr(self, "layer{}".format(i))(x) + y #ccp
        y = getattr(self, "layer{}".format(self.degree))(y)
        return y

def make_harmonics(opt):
    # t = np.random.uniform(0, 1, size=(opt.N, opt.d))
    size = opt.N
    n = opt.d  # or any positive integer
    x = np.random.normal(size=(size, n))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    t = x
    # neta = np.random.uniform(0, 1, size=(len(opt.A), opt.d))
    size = len(opt.A)
    n = opt.d  # or any positive integer
    net = np.random.normal(size=(size, n))
    net /= np.linalg.norm(net, axis=1)[:, np.newaxis]

    neta = net
    yt = reduce(lambda a, b: a + b,
                [Ai * eval_gegenbauer(ki, ((opt.d - 1) / 2), np.inner(t, ni)) for ki, Ai, ni in
                 zip(opt.K, opt.A, neta)])

    return t, yt, neta



def get_residual_projections(model, t, yt, neta, opt):
    # t = np.random.uniform(0, 1, size=(opt.N, opt.d))
    # neta = np.random.uniform(0, 1, size=(len(opt.A), opt.d))
    x = t
    if (opt.CUDA):
        x = t.clone().detach().cpu().numpy()
        # yt = yt.clone().detach().cpu().numpy()

    vts = [eval_gegenbauer(ki, ((opt.d - 1) / 2), np.inner(x, ni)) for ki, ni in zip(opt.K, neta)]


    scaling = opt.THETA * np.sqrt(2.0 / opt.WIDTH)
    if (opt.CUDA):torch.tensor(scaling).float().cuda()
    preds = scaling * model(t)
    residuals = (yt - preds).clone().detach().cpu().numpy()
    # print(residuals[:, 0].shape, vts[0].shape)
    a_ks = [np.inner(v / np.sqrt(x.shape[0]), residuals[:, 0]) for v in vts]
    # print(len(a_ks))
    return a_ks

def to_torch_dataset_1d(opt, t, yt, neta):
    t = torch.from_numpy(t).view(-1, opt.d).float()
    yt = torch.from_numpy(yt).view(-1, 1).float()
    if opt.CUDA:
        t = t.cuda()
        yt = yt.cuda()
    return t, yt, neta

def get_residuals(opt, frames):
    k = len(opt.K)
    ak_s = [[] for i in range(k)]

    for i in range(len(frames[0])):
        for j in range(k):
            ak_s[j].append(np.absolute(frames[0][i].residual_aks[j]))

    return ak_s




