import argparse
import pdb

import torch
import torch.nn as nn
import os
import numpy as np
import time
from utils import mnist_db
import logging
import pdb
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print('Using PyTorch version:', torch.__version__, ' Device:', device)

def train():
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)*args._scale_train
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,  target.float().reshape(-1,1))
        loss.backward()
        optimizer.step()

def validate(loader,scale_test,scale_train,sqrt):
    global is_best, best_prec
    model.eval()
    val_loss=0

    rounded_correct = 0
    rounded_incorrect = 0
    floor_ceil_correct = 0
    floor_ceil_incorrect = 0
    leeway_correct = 0
    leeway_incorrect = 0

    for data, target in loader:
        if sqrt and scale_test==scale_train:
            data = torch.sqrt(data.to(device)) * scale_test
        else:
            data = data.to(device)*scale_test
        target = target.to(device).reshape(-1,1).float()
        output = model(data).detach()
        val_loss += criterion(output, target).data.item()

        #output=output.float().data

        rounded_prediction = torch.round(output)
        floor_prediction = torch.floor(output)
        ceiling_prediction = torch.ceil(output)
        # Rounded to the nearest integer

        rounded_correct += rounded_prediction.eq(target).sum()

        floor_ceil_correct+=(floor_prediction.eq(target) | ceiling_prediction.eq(target)).sum()
        # if (floor_prediction == y_test[i]) or (ceiling_prediction == y_test[i]):
        #     floor_ceil_correct += 1
        # else:
        #     floor_ceil_incorrect += 1
        # Leeway of 1
        leeway_correct += torch.le(torch.abs(rounded_prediction -target),torch.ones(target.shape).to(device)).sum()

    round_acc = rounded_correct.item()/len(loader.dataset)
    floor_ceil_acc = floor_ceil_correct.item()/len(loader.dataset)
    leeway_acc = leeway_correct.item()/len(loader.dataset)

    is_best = round_acc > best_prec
    if is_best:
        best_prec = round_acc

    val_loss /= len(loader)
    return round_acc,floor_ceil_acc,leeway_acc,val_loss

def save_checkpoint(state, is_best, filename):
    if not is_best:
        return
    torch.save(state, filename)

class mlp(nn.Module):
    def __init__(self,degree,input_dim=28 * 28,hidden_dim=50,output_dim=10,biaspre=True,biaslast=True,batchnorm=False):
        super(mlp, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim = output_dim
        self.degree=degree
        self.biaspre=biaspre
        self.biaslast=biaslast
        self.batchnorm=batchnorm
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim,self.biaspre)])
        if self.batchnorm:
            self.bns= nn.ModuleList([nn.BatchNorm1d(self.hidden_dim)])

        for layer in range(self.degree - 2):
            self.linears.append(nn.Linear(self.hidden_dim, self.hidden_dim,self.biaspre))
            if self.batchnorm:
                self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        self.linears.append(nn.Linear(self.hidden_dim, int(self.output_dim),self.biaslast))

    def forward(self, z):
        x = z.view(-1, self.input_dim)
        for i in range(self.degree-1):

            x = self.linears[i](x)
            if self.batchnorm:
                x=self.bns[i](x)
            x = nn.ReLU()(x)
        x = self.linears[self.degree-1](x)
        return x


class pinet(nn.Module):
    def __init__(self,degree,use_act,input_dim=28 * 28,hidden_dim=50,output_dim=10,biaspre=True,biaslast=True,batchnorm=False):
        super(pinet, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim = output_dim
        self.degree=degree
        self.act = nn.ReLU() if use_act==True else lambda x:x
        self.channel_list = [self.input_dim]+[self.hidden_dim]*(self.degree)+[self.output_dim]
        self.biaspre=biaspre
        self.biaslast=biaslast
        self.batchnorm=batchnorm
        setattr(self, 'l{}_0'.format(0), nn.Linear(self.channel_list[0],self.channel_list[1],bias=self.biaspre))
        if self.batchnorm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.channel_list[1])])
        for i in range(1,self.degree):
            setattr(self, 'l{}_0'.format(i),nn.Linear(self.channel_list[i], self.channel_list[i+1],bias=self.biaspre))
            if self.batchnorm:
                self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        setattr(self, 'l{}_rgb', nn.Linear(self.channel_list[-2], self.channel_list[-1], bias=self.biaslast))

    def forward(self, z):
        z= z.view(-1, self.input_dim)
        h = z + 0
        h = getattr(self, 'l{}_0'.format(0))(h)
        if self.batchnorm:
            h = self.bns[0](h)
        h = self.act(h)

        for cursor in range(1, self.degree):
            shortcutskip = h
            h = getattr(self, 'l{}_0'.format(cursor))(h)
            if self.batchnorm:
                h = self.bns[cursor](h)
            h = self.act(h)
            h = h*(nn.AdaptiveAvgPool1d(h.shape[1])(z.unsqueeze(1)).squeeze(1)) + shortcutskip
        h = getattr(self, 'l{}_rgb')(h)
        return h

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str,help='"mlp"   "pinet"')
    parser.add_argument('--dataname', nargs='+',default=["fold2","fold4","fold6"])
    parser.add_argument('--_degree', type=int,default=4)
    parser.add_argument('--_class', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int,default=256, help='')
    parser.add_argument('--_scale_test', nargs='+',default=[], type=float)
    parser.add_argument('--_scale_train',default=1, type=float)
    parser.add_argument('--use_act',default=True,type=int)
    parser.add_argument('--batchnorm', default=False, type=int)
    parser.add_argument('--sqrt', default=False, type=int)
    parser.add_argument('--biaspre',default=True,type=int)
    parser.add_argument('--biaslast',default=True,type=int)
    parser.add_argument('--opt', default="sgd", type=str)
    args = parser.parse_args()


    files_dir = '%s/%s-%s-degree%s-class%s-width%s-testr%s-biaspre%d-biaslast%d-act%d-bn%d-%s' \
                % ("resultsaddition",args.modelname,args.opt,args._degree, args._class, args.hidden_dim
                   ,'_'.join([str(elem) for elem in args._scale_test]),args.biaspre,args.biaslast,args.use_act,args.batchnorm,time.strftime('%Y-%m-%d-%H-%M-%S')[5:])
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
    logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M',
                        filename="%s/%s" %(files_dir,"out.log"), filemode='w+')

    accuracy_runs_ID=[]
    accuracy_runs_OOD=[]

    if args.modelname !="pinet":
        args._degree+=1
    batch_size = 128
    epochs = 100

    for run in range(len(args.dataname)):
        print("\n--------run%d--------"%(run))
        logging.info("\n--------run%d--------"%(run))
        input_dim = 28 * 28 * 2
        train_loader, validation_loader = mnist_db(args.dataname[run],batch_size=batch_size)
        if args.modelname == "pinet":
            model = pinet(degree=args._degree,output_dim=args._class,input_dim=input_dim,hidden_dim=args.hidden_dim, biaspre= args.biaspre,biaslast=args.biaslast,use_act=args.use_act).to(device)
        elif args.modelname =="mlp":
            model = mlp(degree=args._degree, output_dim=args._class,input_dim=input_dim,hidden_dim=args.hidden_dim, biaspre= args.biaspre,biaslast=args.biaslast).to(device)
        print("Parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))
        logging.info("Parameters:%.d"%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        if args.opt=="sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-5, momentum=0.9, nesterov=True)
        elif args.opt=="ada":
            optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-08, weight_decay=0.0)
        else:
            raise("error")
        criterion = nn.MSELoss()
        best_prec = 0.0
        for epoch in range(1, epochs + 1):
            train()
            if (epoch-1) %10==0:
                round_acc, floor_ceil_acc, leeway_acc, val_loss= validate(validation_loader,args._scale_train,args._scale_train,args.sqrt)
                outstring = 'Epoch {:d}, Round: {:.4f}, FloorCeil: {:.4f}, Leeway: {:.4f}, Val loss: {:.4f}'.\
                    format(epoch,round_acc, floor_ceil_acc, leeway_acc,val_loss, )
                print(outstring)
                logging.info(outstring)

        filename = os.path.join(files_dir, 'run%d.pth.tar' % (run))
        is_best=True
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_prec': best_prec}, is_best, filename)
        # checkpoint = torch.load(filename, map_location=device)
        # model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        round_acc, floor_ceil_acc, leeway_acc, loss= validate(train_loader, args._scale_train, args._scale_train, args.sqrt)
        outstring = 'ID loss: {:.4f}, Round: {:.4f}, FloorCeil: {:.4f}, Leeway: {:.4f}'.format(loss, round_acc, floor_ceil_acc, leeway_acc)
        print(outstring)
        logging.info(outstring)
        accuracy_runs_ID.append([round_acc, floor_ceil_acc, leeway_acc])

        round_acc, floor_ceil_acc, leeway_acc, loss = validate(validation_loader, args._scale_train, args._scale_train, args.sqrt)
        outstring = 'OOD loss: {:.4f}, Round: {:.4f}, FloorCeil: {:.4f}, Leeway: {:.4f}'.format(loss, round_acc, floor_ceil_acc, leeway_acc)
        print(outstring)
        logging.info(outstring)
        accuracy_runs_OOD.append([round_acc, floor_ceil_acc, leeway_acc])
    print("--------Final result--------")
    logging.info("--------Final result--------")

    accuracy_runs_ID = np.array(accuracy_runs_ID)
    outstr = ("ID : Round %.4f + %.4f,  FloorCeil %.4f + %.4f,  Leeway %.4f + %.4f") % \
                                                               (np.mean(accuracy_runs_ID[:,0]), np.std(accuracy_runs_ID[:,0]),
                                                               np.mean(accuracy_runs_ID[:,1]), np.std(accuracy_runs_ID[:,1]),
                                                               np.mean(accuracy_runs_ID[:,2]), np.std(accuracy_runs_ID[:,2]),)
    print (outstr)
    logging.info(outstr)

    accuracy_runs_OOD = np.array(accuracy_runs_OOD)
    outstr = ("OOD : Round %.4f + %.4f,  FloorCeil %.4f + %.4f,  Leeway %.4f + %.4f") % \
             (np.mean(accuracy_runs_OOD[:, 0]), np.std(accuracy_runs_OOD[:, 0]),
              np.mean(accuracy_runs_OOD[:, 1]), np.std(accuracy_runs_OOD[:, 1]),
              np.mean(accuracy_runs_OOD[:, 2]), np.std(accuracy_runs_OOD[:, 2]),)
    print(outstr)
    logging.info(outstr)