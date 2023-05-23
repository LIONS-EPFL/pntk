import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import *
from functools import partial
import pdb
DEFAULT_MODE = -1
DEFAULT_PAIR = (-1,-1)
DEFAULT_IND = -1

def square(x):
    return x ** 2

def mape(y_pred, y):
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
    return 100.0 * torch.median(e)

cls_criterion = torch.nn.CrossEntropyLoss()
mse_criterion = torch.nn.MSELoss()
lossfun = {'cls': cls_criterion, 'reg': mse_criterion, 'mape': mape}
#actfun = {'sin': torch.sin, 'square': square, 'tanh': F.tanh, 'exp': torch.exp, 'log':torch.log, 'relu': F.relu, 'gelu': F.gelu, 'sigmoid': F.sigmoid}
actfun = { 'relu': F.relu, 'noact': lambda x:x}


class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name
        self.loss_fn = args.loss_fn
        self.activation = args.activation
        self.actfunc = actfun[self.activation]
        
    def train_(self, input_nodes, label):
        self.optimizer.zero_grad()
        output = self(input_nodes)
        
        if self.loss_fn != 'cls':

            loss = lossfun[self.loss_fn](output.view(label.shape), label)
            
            loss.backward()
            self.optimizer.step()
            
            mape_loss = lossfun['mape'](output.view(label.shape), label)
            return 0, loss.cpu(), mape_loss.cpu()
        else:
            loss = lossfun[self.loss_fn](output, label)
            
            loss.backward()
            self.optimizer.step()
            
            pred = output.data.max(1)[1]
            correct = pred.eq(label.data).cpu().sum()
            accuracy = correct.to(dtype=torch.float) * 100. / len(label)
            
        return accuracy, loss.cpu(), 0
        
    def test_(self, input_nodes, label, print_info=False):
        with torch.no_grad():
            output = self(input_nodes)
            if print_info:
                print(output.view(-1), label)

            if self.loss_fn != 'cls':
                loss = lossfun[self.loss_fn](output.view(label.shape), label)
                mape_loss = lossfun['mape'](output.view(label.shape), label)
                return 0, loss.cpu(), mape_loss.cpu(),output
            else:
                loss = lossfun[self.loss_fn](output, label)
                pred = output.data.max(1)[1]
                correct_ind = pred.eq(label.data).cpu()
                correct = pred.eq(label.data).cpu().sum()
                accuracy = correct.to(dtype=torch.float) * 100. / len(label)
                return accuracy, loss.cpu(), 0,output

    def pred_(self, input_nodes):
        with torch.no_grad():
            output = self(input_nodes)
            pred = output.data.max(1)[1]
            return pred

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class pinets(BasicModel):
    def __init__(self, args):
        super(pinets, self).__init__(args, 'pinets')
        self.degree = args.mlp_layer
        self.input_dim, self.hidden_dim, self.output_dim = args.input_dim, args.hidden_dim, args.output_dim
        self.option = args.option
        self.actfunc = nn.ReLU() if self.activation == "relu" else lambda x:x
        self.channel_list = [self.input_dim]+[self.hidden_dim]*(self.degree)+[self.output_dim]
        setattr(self, 'l{}_0'.format(0), nn.Linear(self.channel_list[0],self.channel_list[1],bias=False))
        for i in range(1,self.degree):
            ##in_channels = self.channel_list[i]
            out_channels = self.channel_list[i+1]
            setattr(self, 'locz{}'.format(i), nn.Linear(self.channel_list[0], out_channels,bias=False))
        setattr(self, 'l{}_rgb',nn.Linear(self.channel_list[-2],self.channel_list[-1],bias=False))

        if self.option == 'A':
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.option == 'B':
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=sqrt(2))

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr, weight_decay=args.decay)
        # self.bns = nn.BatchNorm1d(self.output_dim)

    def forward(self, z):
        h = z + 0
        h = getattr(self, 'l{}_0'.format(0))(h)
        h = self.actfunc(h)
        for cursor in range(1, self.degree):
            z1 = getattr(self, 'locz{}'.format(cursor))(z)
            z1 = self.actfunc(z1)
            h = z1*h
        h = getattr(self, 'l{}_rgb')(h)
        # h = self.bns(h)
        return h


class pinetlarge(BasicModel):
    def __init__(self, args):
        super(pinetlarge, self).__init__(args, 'pinetlarge')
        layer_d=[1,10,10,10,10,10,10,10,10,10,10,10,10,10]
        #layer_d = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        use_bn=False
        n_out= 2
        mult_lat=True
        learn_init=False
        z_mult=1 
        mult_until=-1
        dim_z=1 #  yml :2!!!
        use_localz=True
        concat_inj=False
        allow_repeat=False
        normalize_preinject=False
        use_out_tanh=False
        use_activ=False

        Linear = torch.nn.Linear
        # # Define dim_z.
        self.repeat_z = 0
        if dim_z is None or dim_z == 0:
            self.dim_z = dim_z = layer_d[0]
        else:
            self.dim_z = dim_z
            if dim_z == 1 and allow_repeat:
                # # how many times we need to repeat z. Assumes a constant
                # # number of dimensions in ALL the layers.
                self.repeat_z = layer_d[1]
        self.use_bn = use_bn
        self.use_activ = use_activ
        self.learn_init = learn_init
        # # z_mult: Global transformation layers for all z.
        self.z_mult = z_mult
        self.mult_lat = mult_lat
        assert mult_lat, 'Non-injected version not implemented here.'
        self.use_localz = use_localz
        # # concat_inj: If True, concatenate the injection instead of multiplying.
        self.concat_inj = concat_inj
        factor = 1 if not self.concat_inj else 2
        self.normalize_preinject = normalize_preinject
        self.use_out_tanh = use_out_tanh and not self.use_activ
        if self.use_out_tanh:
            print('Using the output tanh in the synthetic case!')

        self.n_l = n_l = len(layer_d)
        # # set the mult_until, i.e. till which layer to multiply the latent representations.
        self.mult_until = mult_until if mult_until > 0 else n_l + 1
        # with self.init_scope()


        if learn_init:
            # # Learnable param independent of the batch size, i.e. common for each sample.
            self.x = chainer.Parameter(initializer=self.xp.random.randn(1, self.dim_z), shape=[1, self.dim_z])
        # # Global affine transformations of z.
        if z_mult is not None and z_mult > 0:
            for l in range(1, z_mult + 1):
                setattr(self, 'zgL{}'.format(l), Linear(self.dim_z, self.dim_z))
        # # iterate over all layers (till the last) and save in self.
        for l in range(1, n_l):
            # # save the self.layer.
            if l == 1:
                setattr(self, 'l{}'.format(l),
                        Linear(self.concat_inj * self.dim_z + layer_d[l - 1], layer_d[l]))
            else:
                setattr(self, 'l{}'.format(l), Linear(factor * layer_d[l - 1], layer_d[l]))

        # # save the last layer.
        setattr(self, 'l{}'.format(n_l), Linear(factor * layer_d[-1], n_out))

        # # set the batch normalization normalization if use_bn is true.
        if use_bn:
            bn1 = partial(torch.nn.BatchNorm1d, use_gamma=True, use_beta=False)
            # # set the batch norm for the first layer.
            setattr(self, 'bn{}'.format(1), bn1(self.dim_z))
            for l in range(2, self.n_l + 1):
                sz = getattr(self, 'l{}'.format(l - 1)).out_size
                # # set the batch norm for the layer.
                setattr(self, 'bn{}'.format(l), bn1(sz))

        if self.mult_lat and use_localz:
            for l in range(1, min(self.mult_until, self.n_l)):
                # # : potential problem in case of repeat_z > 0 and use_localz=1.
                # # 16/9: It had a factor * layer_d[l] before. However, for sin3D with concat,
                # # it works without the factor here (makes sense to have only layer_d[l]).
                setattr(self, 'locz{}'.format(l + 1), torch.nn.Linear(dim_z, layer_d[l]))

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr, weight_decay=args.decay)

    def forward(self, z=None, **kwargs):
        # if z is None:
        #     # # sample both z and z0 (the latter used as the 'bias' of Taylor term).
        #     z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        #     if self.add_z0:
        #         z0 = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if len(list(z.shape))==3:
            z = z.squeeze(2)
        batchsize = z.shape[0]
        activ = F.relu if self.use_activ else lambda x: x
        if self.learn_init:
            h = torch.reshape(self.x, (1, self.dim_z))
            # # replicate batch times.
            h = torch.repeat(h, batchsize, axis=0)
            z = torch.reshape(z, (z.shape[0], z.shape[1]))
        else:
            h = torch.reshape(z, (z.shape[0], z.shape[1]))
        if self.z_mult is not None and self.z_mult > 0:
            for l in range(1, self.z_mult + 1):
                z = getattr(self, 'zgL{}'.format(l))(z)
        if self.repeat_z:
            # repeat constant number of times.
            z =z.repeat(1,self.repeat_z) #F.repeat(z, self.repeat_z, axis=1)
        # # loop over the layers and get the layers along with the
        # # normalizations per layer.
        for layer in range(1, self.n_l + 1):
            # # step 1: normalize representations.
            if self.use_bn:
                h = getattr(self, 'bn{}'.format(layer))(h)
            # # step 2: element-wise mult. + addition.
            if layer < self.mult_until:
                if self.use_localz and layer > 1:
                    # # apply local transformation.
                    z1 = getattr(self, 'locz{}'.format(layer))(z)
                    if self.normalize_preinject:
                        print(layer, z1.shape, h.shape)
                        z1 /= F.sqrt(F.mean(z1 * z1, axis=1, keepdims=True) + 1e-8)
                    if self.concat_inj:
                        h = F.concat((h, z1), axis=1)
                    else:
                        h = h * z1
                else:
                    if self.concat_inj:
                        h = F.concat((h, z), axis=1)
                    else:
                        h = h * z
            # # step 3: FC + activation.
            h = activ(getattr(self, 'l{}'.format(layer))(h))
        if self.use_out_tanh:
            h = F.tanh(h)
        if len(h.shape) == 2:
            h = torch.reshape(h, (h.shape[0], h.shape[1], 1, 1))
        return h

