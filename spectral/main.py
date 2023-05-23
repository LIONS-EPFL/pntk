import pdb
import yaml
import torch
import torch.nn as nn
import argparse
import os
import pickle
import numpy as np
import time
import logging
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from argparse import Namespace
from utils import pi_model_ntk,make_harmonics,get_residual_projections,to_torch_dataset_1d
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


def make_model(opt):
    def weights_init(m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 1)
    model = pi_model_ntk(opt)
    if opt.CUDA:
        model = model.cuda()
    outstring="Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(outstring)
    logging.info(outstring)
    model.apply(weights_init) 
    return model

def train_model(opt, model, input_, target, neta):
    # Build loss
    loss_fn = nn.MSELoss()
    # Build optim
    optim = torch.optim.SGD(model.parameters(), lr=opt.LR)
    # Rec
    frames = []
    model.train()
    # To cuda
    if opt.CUDA:
        input_ = input_.cuda()
        target = target.cuda()
    # Loop!
    for iter_num in range(opt.NUM_ITER):
        x = input_
        yt = target.view(-1, 1)
        optim.zero_grad()
        scaling = opt.THETA*np.sqrt(2.0/opt.WIDTH)
        if(opt.CUDA):torch.tensor(scaling).float().cuda()
        y = scaling*model(x)
        loss = loss_fn(y, yt)
        loss.backward()
        optim.step()
        residual =  get_residual_projections(model, input_, yt, neta, opt)
        if (iter_num) % opt.PRINTLOSS_INTERVAL == 0:  # if (iter_num+1) % (opt.NUM_ITER // 10) == 0:
            lossclone =loss.clone().detach().cpu().numpy()
            residual_str = "  ".join([str(round(i,4)) for i in residual])
            outstring="iter=%d Loss=%.4f, res=%s"%(iter_num,lossclone,residual_str)
            print(outstring)
            logging.info(outstring)
            if torch.isnan(loss):
                print("NAN loss, error!!!!!!!!!!!!")
                logging.info("NAN loss, error!!!!!!!!!!!!")
                break
        if iter_num % opt.REC_FRQ == 0:
            frames.append(Namespace(iter_num=iter_num,
                                    #prediction=y.data.cpu().numpy(),
                                    loss=loss.item(),
                                    residual_aks =residual))
                                    #spectral_norms=spectral_norm(model)))
    # Done
    model.eval()
    return frames

def go(opt, x, y, neta, repeats=1):
    all_frames = []
    for _ in range(repeats):
        # Sample random phase
        # opt.PHI = [np.random.rand() for _ in opt.K]
        # Generate data
        # x, y, neta = to_torch_dataset_1d(opt, *make_harmonics(opt))
        # Make model
        model = make_model(opt)
        # Train
        frames = train_model(opt, model, x, y, neta)
        all_frames.append(frames)
        print('', end='\n')
    return all_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', type=int)
    parser.add_argument('--width', type=int,default = 32768)
    parser.add_argument('--bias',default=False,type=int)
    parser.add_argument('--resultdir',type=str)
    parser.add_argument('--lr',default=0.0016,type=float)

    args = parser.parse_args()


    opt = Namespace()
    opt.N = 1000
    opt.d = 10
    #opt.A = [i*1e-4 for i in opt.A]

    opt.A = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    opt.K = [1, 3, 4, 5, 8, 12]
    opt.CUDA = True if torch.cuda.is_available() else False
    print("Device: ",opt.CUDA)
    if opt.CUDA == False:
        opt.WIDTH = args.width
        opt.REC_FRQ =  2
        opt.NUM_ITER = 30000 #4
        opt.PRINTLOSS_INTERVAL = 30
    else:
        opt.WIDTH = args.width
        opt.REC_FRQ = 10
        opt.NUM_ITER = 30000
        opt.PRINTLOSS_INTERVAL = opt.NUM_ITER//10

    args.lr =float(args.lr)
    opt.LR = args.lr
    opt.THETA = 1
    opt.degree = args.degree
    opt.bias =args.bias
    x, y, neta = to_torch_dataset_1d(opt, *make_harmonics(opt))
    # print(x.shape, y.shape)
    np.mean(np.absolute(y.cpu().numpy()))
    files_dir = '%s/degree%d' \
                % (args.resultdir,opt.degree)
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
    logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M',
                        filename="%s/%s" % (files_dir, "out.log"), filemode='w+')
    frames = go(opt, x, y, neta)
    # plot_residuals(opt, get_residuals(opt, frames))
    with open(os.path.join(files_dir,'setup.yaml'), 'w') as file: documents = yaml.dump(vars(opt), file)
    frames_array = []
    for i in frames[0]:
        cur = [i.iter_num,i.loss] + i.residual_aks
        frames_array.append(cur)
    frames_array = np.array(frames_array)
    with open(os.path.join(files_dir,'log.pkl'), 'wb') as f:
        #pickle.dump((X_train, y_train, X_test, y_test), f)
        pickle.dump(frames_array, f)
