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

dataset_name = "./data/non-linear/helix/helix_xdim1_item1_trainscube_testscube_signno_fix1_testr2.0_trainr1.0_valr1.0_ntrain2000_nval1000_ntest2000_Ar1.0_br0.0.pickle"
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
y = label.to(args.device).numpy().squeeze()
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
y = label.to(args.device).numpy() .squeeze()
X_OOD, y_OOD = [], []
for i in range(len(y)):
    X_OOD.append(X[i])
    y_OOD.append(y[i])
X_OOD =np.array(X_OOD)
y_OOD=np.array(y_OOD)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_OOD[:,0], y_OOD[:,1], X_OOD,c="grey",label="OOD", alpha=0.1)
ax.scatter(y_inD[:,0], y_inD[:,1], X_inD,c="blue",label="InD") #xs, ys, zs : shape:(100,)
r = 3
ax.set_xlim(-1*r,r)
ax.set_ylim(-1*r, r)
ax.set_zlim(-r, r)
plt.legend()
plt.show()