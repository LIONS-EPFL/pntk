# Extrapolation and Spectral Bias of Neural Nets with Hadamard Product: a Polynomial Net Study

Code for the Neurips'22 paper called "[Extrapolation and Spectral Bias of Neural Nets with Hadamard Product: a Polynomial Net Study](https://arxiv.org/abs/2209.07736)".

The repo includes the source code for various experiments in the paper.


## Requirement
To run the code, please create a conda environment and install the following packages
```
conda create -n extra python=3.7.7

conda activate extra

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install pyyaml

pip install scipy==1.1.0

pip install matplotlib

pip install seaborn
```

## Browsing the folder and files
The folder is organized as follows:

`extra/`:  contains the code for the experiment on extrapolation

`spectral`:  contains the code for the experiment on spectral bias

`vaec`: contains the code for the experiment on vaec dataset


## To run experiment on extrapolation
```
cd extra
sh script_extra.sh
```

## For the experiment on VAEC dataset
Firstly, download the dataset base on the instruction in the [original repo](https://github.com/taylorwwebb/learning_representations_that_support_extrapolation). Then run the followng command to generate data: 
```
cd extra/dset_gen
sh generated_data.sh
```
Training:
```
cd script
sh context_norm_scale_extrap.sh
```

## To run the experiment on spectral bias
To train the network on MNIST dataset: run
```
cd spectral
sh script_spectral.sh
```
Next, to visualzie the result, run 
```
python eval.py
```

## Reference:

https://github.com/jinglingli/nn-extrapolate
https://github.com/taylorwwebb/learning_representations_that_support_extrapolation
https://github.com/grigorisg9gr/polynomial_nets_for_conditional_generation/tree/master/conditional_generation_with_gan




## Cite as:
If you use this code, please cite 
```
@inproceedings{
wu2022extrapolation,
title={Extrapolation and Spectral Bias of Neural Nets with Hadamard Product: a Polynomial Net Study},
author={Yongtao Wu and Zhenyu Zhu and Fanghui Liu and Grigorios G Chrysos and Volkan Cevher},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
}
```
