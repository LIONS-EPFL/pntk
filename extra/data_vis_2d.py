import pdb
import matplotlib.pyplot as plt
from main_cos import load_data, cvt_data_axis, tensor_data
from argparse import ArgumentParser
import numpy as np
parser = ArgumentParser()
args = parser.parse_args()
args.device = "cpu"
args.loss_fn = "None"

def get_range(dataset_name):
    line = dataset_name.split('/')[-1]
    ks = {}
    for item in line.split('_'):
        if 'trainr' in item:
            r = float(item.replace('trainr', ''))
            return r

dataset_name = "./data/linear_miss_direction/square_xdim2_item1_trainssphere_testssphere_sign2p_fix1_testr5.0_trainr2.0_valr2.0_ntrain2000_nval1000_ntest8000_Ar10.0_br0.0.pickle"
datasets = {
    'train': load_data(dataset_name, 0),
    'dev': load_data(dataset_name, 1),
    'test': load_data(dataset_name, 2)
}
dim = datasets['train'][0][0].shape[0]
r = get_range(dataset_name)


# read data
dataset = datasets['train']
args.batch_size = len(dataset)
dataset = cvt_data_axis(dataset)
input_nodes, label = tensor_data(dataset, 0, args)
data = input_nodes.data.view((-1, dim))
X = data.numpy()
y = label.to(args.device).numpy()
X_inD, y_inD= [], []
for i in range(len(y)):
    X_inD.append(X[i])
    y_inD.append(y[i])
X_inD =np.array(X_inD)
y_inD=np.array(y_inD)

dataset = datasets['test']
args.batch_size = len(dataset)
dataset = cvt_data_axis(dataset)
input_nodes, label = tensor_data(dataset, 0, args)
data = input_nodes.data.view((-1, dim))
X = data.numpy()
y = label.to(args.device).numpy()
X_OOD, y_OOD = [], []
for i in range(len(y)):
    X_OOD.append(X[i])
    y_OOD.append(y[i])
X_OOD =np.array(X_OOD)
y_OOD=np.array(y_OOD)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_OOD[:,0], X_OOD[:,1], y_OOD,c="grey",label="OOD", alpha=0.1)
ax.scatter(X_inD[:,0], X_inD[:,1], y_inD,c="blue",label="InD") #xs, ys, zs : shape:(100,)
r = 5
ax.set_xlim(-1*np.sqrt(r),np.sqrt(r))
ax.set_ylim(-1*np.sqrt(r), np.sqrt(r))
ax.set_zlim(0, r)
plt.legend()
plt.show()