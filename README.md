# Depth-Estimation-on-Resource-Limited-Devices



---
### Table of Contents

- [Description](#description)
- [Install and Dependencies](#install-and-dependencies)
- [How To Use](#how-to-use)
- [Author Info](#author-info)
--- 
## Description

```diff
@@ Novel Depth Estimation Model for Resource-limited Devies @@
@@ Bachelor's thesis project. HSE, Moscow. 2021 @@
```
**Novel Depth Estimation Model for Resource-limited Devies**
*Bachelor's thesis project. HSE, Moscow. 2021*

SOTA Depth Estimation deep learning models usually require high computational resource in order to operate. 
Neural-Architecture-Search (NAS) is a technique for automating the designing of artificial neural networks without assistance of field expertise. 

We have used NAS ([ProxylessNAS](https://arxiv.org/abs/1812.00332)) approach to design a novel depth estimation model with high latency and accuracy. 


#### Table of Contents
- [dependencies.txt](dependencies.txt) : Required packages to run
- [mymain.py](mymain.py) : To start neural architecture search process
- [mymodel.py](mymodel.py) : Base model of the search space
- [ops.py](ops.py) : Operations (convolution, upconvolution, etc.)
- [mynyu.py](mynuy.py) : [NYUDepth](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset and dataloader
- [transforms.py](transforms.py) : Data transformation utility functions
- [myutils.py](myutils.py) : Utility functions

--
## Install And Dependencies

> some text
#### Dependencies
- Neural Network Intelligence 

--
## How To Use

```bash
  git clone https://github.com/tombuuz/Depthmap-Resource-Limited
```

Install required packages
```bash
  conda create -n myenv --file dependencies.txt
```

To install [NNI](https://github.com/microsoft/nni)
```bash
  pip install nni
```
or 
```bash
  git clone https://github.com/microsoft/nni 
```


--
## Author Info

#### Author: Dagvanorov Lkhagvajav 
#### Email: Ldagvanorov@gmail.com
#### Scientific Supervisor: [lya Makarov](https://www.hse.ru/en/staff/iamakarov)


