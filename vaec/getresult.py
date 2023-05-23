import pdb
import os
import numpy as np

def parse_log(logpath):
    cur =0
    for line in open(logpath, "r"):
        if cur==0:
            cur+=1
            pass
        else:
            return float(line)

dirs="eval"
res =np.zeros((8,6)) #[[]*8] #
#restest= [[]*8]

for idx,i in enumerate(os.listdir(dirs)):
    for j in os.listdir(os.path.join(dirs, i)):
        acc = parse_log(os.path.join(dirs, i, j))
        if "train" in j:
            res[idx][0]=acc
        if "test" in j:
            res[idx][int(j[4])]=acc
print("all")
print(res)
print("mean and std")
mean=np.around(res.mean(axis=0),1)
std=np.around(res.std(axis=0),1)
for i in range(6):
    print(mean[i],std[i])

