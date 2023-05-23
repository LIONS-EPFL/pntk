import pdb
import yaml
import os
import pickle
from argparse import Namespace
from utils import get_residuals
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def get_opt_frame (file_dir):
    with open(os.path.normpath(os.path.join(file_dir,'setup.yaml')), 'rb') as file:
        opt = Namespace(**yaml.safe_load(file))

    with open(os.path.normpath(os.path.join(file_dir,'log.pkl')), 'rb') as f:
        #pickle.dump((X_train, y_train, X_test, y_test), f)
        frames_array= pickle.load(f)
    # convert frames_array to frames
    frames=[[]]
    for i in range(frames_array.shape[0]):
        res = Namespace(iter_num=frames_array[i][0],
                  # prediction=y.data.cpu().numpy(),
                  loss=frames_array[i][1],
                  residual_aks=frames_array[i][2:].tolist())
        frames[0].append(res)

    return opt,frames


def plot_residuals(opt, ak_s):
    plt.figure(figsize=(8, 6))
    for i in range(len(opt.K)):
        plt.plot(np.arange(len(ak_s[i]))*opt.REC_FRQ, np.log(ak_s[i]), label=str(opt.K[i])+"th harmonic")
    plt.legend(loc="upper right")
    plt.title(opt.mtype + " model")
    plt.show()

def compare_residuals(opt, fr_st, fr_pi):
    k = len(opt.K)
    st = get_residuals(opt, fr_st)
    pi = get_residuals(opt, fr_pi)

    aks_st = []
    aks_pi = []

    for ser in st:
        aks_st.append(np.convolve(ser, np.ones(20) / 20, mode='valid'))

    for ser in pi:
        aks_pi.append(np.convolve(ser, np.ones(20) / 20, mode='valid'))

    rows = (k // 3) if k % 3 == 0 else ((k // 3) + 1)
    # print(rows)
    fig, ax = plt.subplots(nrows=rows, ncols=3, figsize=(18, 12), sharey=True)
    for i in range(k):

        if (rows == 1):
            ax[i % 3].plot(np.arange(len(aks_st[i])) * opt.REC_FRQ, np.log(aks_st[i]), label="model1-kernel")
            ax[i % 3].plot(np.arange(len(aks_pi[i])) * opt.REC_FRQ, np.log(aks_pi[i]), label=r'model2' + "-kernel")#$\Pi$

            ax[i % 3].set_title("k = " + str(opt.K[i]));
            ax[i % 3].legend(loc="upper right");
            ax[i % 3].set_xlabel("Iterations")
            ax[i % 3].set_ylabel("residual projection length (log scale)")
        else:
            ax[int(i / 3), i % 3].plot(np.arange(len(aks_st[i])) * opt.REC_FRQ, np.log(aks_st[i]),
                                       label="model1-kernel")
            ax[int(i / 3), i % 3].plot(np.arange(len(aks_pi[i])) * opt.REC_FRQ, np.log(aks_pi[i]),
                                       label="model2-kernel")
            ax[int(i / 3), i % 3].set_title("k = " + str(opt.K[i]));
            ax[int(i / 3), i % 3].legend(loc="upper right");
            ax[int(i / 3), i % 3].set_xlabel("Iterations")
            ax[int(i / 3), i % 3].set_ylabel("residual projection length (log scale)")

    fig.suptitle(
        r'$f^*(x) = \sum_{k \in \mathcal{K}}0.1P_k(\langle x, \zeta_k \rangle) \rangle), \mathcal{K}= \{1,3,4,5,8,12\}$')
    # fig.suptitle(r'$f^*(x) = \sum_{k \in \mathcal{K}}P_k(\langle x, \zeta_k \rangle) \rangle), \mathcal{K}= \{1,2,4\}$')
    fig.tight_layout(pad=3.0)
    plt.show()

def compare_consol_residuals(opt, fr_list,title):
    k = len(opt.K)
    fig, ax = plt.subplots(nrows=1, ncols=len(fr_list), figsize=(14, 5), sharey = True)#
    for idx,each in enumerate(fr_list):
        aks_st = get_residuals(opt,each)
        for i in range(k):
            ax[idx].plot(np.arange(len(aks_st[i]))*opt.REC_FRQ, np.log(aks_st[i]), label="k = "+ str(opt.K[i]))
            # ax[idx].legend(loc = "upper right");
        ax[idx].set_xlabel("iterations",fontsize=13)
        ax[idx].set_ylabel("Residual projection length (log scale)",fontsize=13)# a_k
        ax[idx].set_title(title[idx], loc="center", fontsize=18)  # ,y=-0.15
    fig.tight_layout(pad=3)
    fig.subplots_adjust(top=0.8,bottom=0.1)##,wspace=0.2,left=0.05,right=0.95
    handles, labels = ax[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels,loc='upper center', prop={'size':18},ncol=6 )#, bbox_transform = plt.gcf().transFigurefancybox=True, shadow=True bbox_to_anchor=(0.5, 1.05),
    plt.show()
    fig.savefig("./harmonic.pdf", bbox_inches='tight')


if __name__ == '__main__':

    files_dir =["result/degree3",
                "result/degree6",
                 "result/degree9",
                ]
    title = ["3-degree","6-degree","9-degree"]