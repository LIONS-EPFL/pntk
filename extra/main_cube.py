import os
import argparse
import pdb
import torch.nn as nn
import pickle
import random
import numpy as np
import torch
import logging
import time
import shutil
import torch.optim as optim
import matplotlib.pyplot as plt
from main_cos import BasicModel
random_seed = 6
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
import numpy as np
from torch.utils.data import Dataset
class Mydataset(Dataset):
    def __init__(self, data, label,labelnoise):
        self.label = label
        self.data = data
        self.noise = labelnoise
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        ynoise=self.noise[idx]
        #x = transforms.ToTensor()(x)
        return x, y,ynoise
noisevar =0
data_path = "finalx3"
def underlying(x):
    y = x**3+x**2-5*x+5
    if noisevar > 0:noisedata =  np.random.normal(0, noisevar, x.shape)
    else: noisedata =  np.zeros(x.shape)
    return y,noisedata
train_r_left,train_r_right = -2.5, 2.5 
test_r_left,test_r_right= -5, 5 

train_number = 200000
pltinterval =  1 #5 #20
saveinterval = 1 #10 #1
test_number= 1000 #2000
train_x= np.random.uniform(train_r_left, train_r_right,(train_number,1,1)) #nois
train_Y,train_noise=underlying(train_x)
test_x= np.random.uniform(test_r_left, test_r_right,(test_number,1,1)) #nois
test_Y,test_noise=underlying(test_x)


best_prec, best_loss, best_model_test_acc, best_model_test_loss, best_model_mape_loss = 0.0, 1e+8 * 1.0, 0.0, 1e+8 * 1.0, 1e+8 * 1.0
is_best = False
best_epoch = 0
def save_checkpoint(state, is_best, epoch, args):
    if not is_best:
        return
    """Saves checkpoint to disk"""
    # data_path = args.data.split("/")[-1]
    directory = args.files_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, 'model_best.pth.tar')
    torch.save(state, filename)


def train(epoch, dataset, args, model):
    model.train()

    running_loss, running_loss_mape = 0.0, 0.0
    accuracys = []
    losses, losses_mape = [], []

    # for batch_idx, (input_nodes, label, noise) in enumerate(dataset):
    #     accuracy, loss, mape_loss = model.train_(input_nodes.float(), label.float() + noise.float())
    train_size = int(dataset[0].shape[0])
    bs =args.batch_size
    batch_runs = max(1, train_size // bs)
    start = time.time()
    for batch_idx in range(batch_runs):
        input_nodes = torch.FloatTensor(dataset[0][args.batch_size*batch_idx:args.batch_size*(batch_idx+1)]).to(args.device)
        noise = torch.FloatTensor(dataset[2][args.batch_size*batch_idx:args.batch_size*(batch_idx+1)]).to(args.device)
        label =  torch.FloatTensor(dataset[1][args.batch_size*batch_idx:args.batch_size*(batch_idx+1)]).to(args.device)

        accuracy, loss, mape_loss = model.train_(input_nodes.float(), label.float() + noise.float())
        running_loss += loss
        running_loss_mape += mape_loss

        accuracys.append(accuracy)
        losses.append(loss)
        losses_mape.append(mape_loss)
    print(batch_idx,time.time() - start)
    avg_accuracy = sum(accuracys) * 1.0 / len(accuracys)
    avg_losses = sum(losses) * 1.0 / len(losses)
    avg_losses_mape = sum(losses_mape) * 1.0 / len(losses_mape)
    print('\n Epoch {:.1f} Train set: accuracy: {:.2f}% \t | loss: {:.7f}  \t | \t mape: {:.7f}'.format(
        epoch,avg_accuracy, avg_losses,avg_losses_mape))
    logging.info(
        '\n Epoch {:.1f} Train set: accuracy: {:.2f}% \t | loss: {:.7f}  \t\t | \t mape: {:.7f}'.format(
            epoch,avg_accuracy, avg_losses,avg_losses_mape))

def test(epoch, dataset, args, model):
    global is_best, best_model_test_acc, best_model_test_loss, best_epoch, best_model_mape_loss

    model.eval()

    accuracys = []
    losses, mape_losses = [], []
    test_size = int(dataset[0].shape[0])
    bs =args.batch_size
    batch_runs = max(1, test_size // bs)
    for batch_idx in range(batch_runs):
        input_nodes = torch.FloatTensor(dataset[0][args.batch_size*batch_idx:args.batch_size*(batch_idx+1)]).to(args.device)
        noise = torch.FloatTensor(dataset[2][args.batch_size*batch_idx:args.batch_size*(batch_idx+1)]).to(args.device)
        label = torch.FloatTensor(dataset[1][args.batch_size*batch_idx:args.batch_size*(batch_idx+1)]).to(args.device)
        accuracy, loss, mape_loss, output = model.test_(input_nodes.float(), label.float() + noise.float())
        accuracys.append(accuracy)
        losses.append(loss)
        mape_losses.append(mape_loss)
    if (epoch - 1) % pltinterval == 0:
        input_nodes = input_nodes.squeeze()
        label = label.squeeze()
        output = output.squeeze()
        noise = noise.squeeze()

        idxsort = input_nodes.argsort()
        input_nodes = input_nodes[idxsort]
        label = label[idxsort]
        output = output[idxsort]
        noise = noise[idxsort]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        #Underlying
        ID_idx_plus1 = (input_nodes <= test_r_right) & (input_nodes >= test_r_left)
        ax.plot(input_nodes[ID_idx_plus1].cpu(), label[ID_idx_plus1].cpu(), c="black", label="Underlying",alpha=0.3)  # xs, ys, zs : shape:(100,)
        if noisevar>0:ax.plot(input_nodes[ID_idx_plus1].cpu(), label[ID_idx_plus1].cpu()+noise[ID_idx_plus1].cpu(), c="grey", label="Underlying", alpha=0.3)
        #ID
        ID_idx = (input_nodes <= train_r_right) & (input_nodes >= train_r_left)
        ax.plot(input_nodes[ID_idx].cpu(), output[ID_idx].cpu(), c="b", label="Pred_ID",alpha=0.8)  # xs, ys, zs : shape:(100,)
        #OOD
        ID_idx_plus1_only =(input_nodes <=test_r_right) & (input_nodes >= train_r_right)
        ax.plot(input_nodes[ID_idx_plus1_only].cpu(), output[ID_idx_plus1_only].cpu(), c="red", label="Pred_OOD",alpha=0.8)  # xs, ys, zs : shape:(100,)
        ID_idx_plus1_only =(input_nodes >= test_r_left) & (input_nodes <=train_r_left)
        ax.plot(input_nodes[ID_idx_plus1_only].cpu(), output[ID_idx_plus1_only].cpu(), c="red", label="Pred_OOD",alpha=0.8)  # xs, ys, zs : shape:(100,)

        fig.savefig(args.files_dir + "/figure%d.png" % (epoch), bbox_inches='tight')
    avg_accuracy = sum(accuracys) * 1.0 / len(accuracys)
    avg_losses = sum(losses) * 1.0 / len(losses)
    avg_losses_mape = sum(mape_losses) * 1.0 / len(mape_losses)

    print('Test set: accuracy: {:.2f}% \t | loss: {:.7f} \t | \t mape: {:.7f} \n'.format(avg_accuracy, avg_losses,
                                                                                         avg_losses_mape))
    logging.info(
        'Test set: accuracy: {:.2f}% \t | loss: {:.7f} \t | \t mape: {:.7f} \n'.format(avg_accuracy, avg_losses,
                                                                                       avg_losses_mape))

    if is_best:
        best_model_test_acc = avg_accuracy
        best_model_test_loss = avg_losses
        best_model_mape_loss = avg_losses_mape
        best_epoch = epoch

    if epoch % 10 == 0:
        print(
            '************ Best model\'s test acc: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {}) ************\n'.format(
                best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch))
        logging.info(
            '************ Best model\'s test acc: {:.2f}%, test loss: {:.7f}, mape: {:.7f} (best model is from epoch {}) ************\n'.format(
                best_model_test_acc, best_model_test_loss, best_model_mape_loss, best_epoch))
        
class pinets(BasicModel):
    def __init__(self, args):
        super(pinets, self).__init__(args, 'pinets')
        self.degree = 3
        self.input_dim, self.hidden_dim, self.output_dim = 1, 1, 1
        self.option = args.option
        self.actfunc =  lambda x: x
        self.channel_list = [self.input_dim] + [self.hidden_dim] * (self.degree) + [self.output_dim]
        setattr(self, 'l{}_0'.format(0),
                nn.Linear(self.channel_list[0], self.channel_list[1], bias=True))
        for i in range(1, self.degree):
            out_channels = self.channel_list[i + 1]
            setattr(self, 'locz{}'.format(i), nn.Linear(self.channel_list[0], out_channels, bias=True))
        setattr(self, 'l{}_rgb', nn.Linear(self.channel_list[-2], self.channel_list[-1], bias=True))

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
            h = z1 * h + h
        h = getattr(self, 'l{}_rgb')(h)
        return h
    
def setup_logs(args):
    file_dir = "results"
    if not args.no_log:
        files_dir = '%s/%s/%s' % (
        file_dir, data_path, time.strftime('%Y-%m-%d-%H-%M-%S')[5:])
        args.files_dir = files_dir
        args.filename = 'lr%s_idim%s_odim%s_bs%s_option%s_epoch%d_seed%d.log' % (
        args.lr, 1, 1, args.batch_size,
        args.option, args.epochs, random_seed)
        if not os.path.exists(args.files_dir):
            os.makedirs(files_dir)
        shutil.copy("main_x3.py", files_dir)
        mode = 'w+'
        if args.resume:
            mode = 'a+'
        logging.basicConfig(format='%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M',
                            filename="%s/%s" % (args.files_dir, args.filename), filemode='w+')
        vars_args = vars(args).copy()
        print(vars_args)
        del vars_args["device"]
        logging.info(vars_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, choices=['A', 'B', 'None'], default='None', help='initialization options')
    # Training settings
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--resume', type=str, help='resume from model stored')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--loss_fn', type=str, choices=['cls', 'reg', 'mape'], default='reg',
                        help='classification or regression loss')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam', help='Adam or SGD')

    # Logging and storage settings
    parser.add_argument('--save_model', action='store_true', default=True, help='flag to store the training models')
    parser.add_argument('--no_log', action='store_true', default=False, help='flag to disable logging of results')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--filename', type=str, default='', help='file to store the training log')
    parser.add_argument('--files_dir', type=str, default='', help='the directory to store trained models logs')


    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    train_datasets = (train_x, train_Y,train_noise) # Mydataset(train_x, train_Y,train_noise)
    test_datasets = (test_x, test_Y,test_noise) #Mydataset(test_x, test_Y,test_noise)

    setup_logs(args)

    model = pinets(args).to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=50, gamma=0.5)

    if args.epochs == 0:
        epoch = 0
        test(epoch, test_datasets, args, model)
        args.epochs = -1

    for epoch in range(1, args.epochs + 1):
        if epoch == 1:
            is_best = True
        train(epoch, train_datasets, args, model)
        test(epoch, test_datasets, args, model)
        scheduler.step()
        # if is_best and args.save_model:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.model,
        #         'args': args,
        #         'state_dict': model.state_dict(),
        #         'best_prec': best_prec,
        #         'best_model_test_acc': best_model_test_acc,
        #         'best_model_test_loss': best_model_test_loss,
        #         'best_model_mape_loss': best_model_mape_loss,
        #         'optimizer': model.optimizer.state_dict(),
        #     }, is_best, epoch, args)
        # if (epoch - 1) % saveinterval == 0:
        #     torch.save(model.state_dict(),os.path.join(args.files_dir, 'model%d.pth.tar'%(epoch)))


    print(
        '************ Best model\'s test acc: {:.2f}%, test loss: {:.7f} throughout training (best model is from epoch {}) ************\n'.format(
            best_model_test_acc, best_model_test_loss, best_epoch))
    logging.info(
        '************ Best model\'s test acc: {:.2f}%, test loss: {:.7f} throughout training (best model is from epoch {}) ************\n'.format(
            best_model_test_acc, best_model_test_loss, best_epoch))

if __name__ == '__main__':
    main()