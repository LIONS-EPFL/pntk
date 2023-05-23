import pdb
import os
import numpy as np
import argparse
def parse_log(logpath):
    cur =0
    for line in open(logpath, "r"):
        if cur==0:
            cur+=1
            pass
        else:
            return float(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=str)
    config = parser.parse_args()

    dirs=config.dirs
    res =np.zeros((8,6)) #[[]*8] #
    #restest= [[]*8]
    for idx,i in enumerate(os.listdir(dirs)):
        for j in os.listdir(os.path.join(dirs, i)):
            acc = parse_log(os.path.join(dirs, i, j))
            if "train" in j:
                res[idx][0]=acc
            if "test" in j:
                if "scale" in j:
                    res[idx][int(j[10])] = acc
                else:
                    res[idx][int(j[4])]=acc
    print("all")
    print(res)
    print("mean and std")
    mean=np.around(res.mean(axis=0),1)
    std=np.around(res.std(axis=0),1)
    for i in range(6):
        print(mean[i],std[i])
