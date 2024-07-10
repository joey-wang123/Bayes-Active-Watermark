## Defense against Model Extraction Attack by Bayesian Active Watermarking (ICML 2024) ##


## Package Requirements
- Pytorch 1.12.1


# Model Extraction Without Any Defense

`cd DFME`

`python3 train.py --model resnet34_8x --dataset cifar10 --ckpt 'path/your/standardCNNmodel'  --device 0 --grad_m 1 --query_budget 20 --log_dir path/to/your/log/dir  --lr_G 1e-4 --student_model resnet18_8x --loss l1;`


# Model Extraction With Proposed Defense

## Step1: Download Pre-trained Model

download the pre-trained model from google drive [here](https://drive.google.com/file/d/1z0IsEm0F5PXnesBqlcckW6feq4kB3HBA/view?usp=sharing)

## Step2: Active Watermarking by Finetuning Pre-trained Model

`cd teacher-train`

### CIFAR10

`python teacher-finetune-AW.py --dataset cifar10 --model resnet34_8x --scale 1.0 --alpha 0.00001 --ckpt 'models/CIFAR10.pth'`

### CIFAR100

`python teacher-finetune-AW.py --dataset cifar100 --model resnet34_8x --scale 0.5 --alpha 0.00001 --ckpt 'models/CIFAR100.pth'`

where 'models/CIFAR10.pth' and 'models/CIFAR100.pth' are the model downloaded in step 1

## Step3: Defense Experiment



`cd DFME`

### CIFAR10

`python3 train_cifar10_AW.py --model resnet34_8x --dataset cifar10 --ckpt 'path/to/your/finetunedmodel'   --device 0 --grad_m 1 --query_budget 20 --log_dir path/to/your/log/dir  --lr_G 1e-4 --student_model resnet18_8x --loss l1;`

### CIFAR100

`python3 train_cifar100_AW.py --model resnet34_8x --dataset cifar100 --ckpt 'path/to/your/finetunedmodel'  --device 0 --grad_m 1 --query_budget 200 --log_dir path/to/your/log/dir --lr_G 1e-4 --student_model  resnet18_8x --loss l1;`

where the 'path/to/your/finetunedmodel' is obtained from step 2

## Citation

If you find our paper or this resource helpful, please consider cite:

```
@inproceedings{wang2024a,
title={Defense against Model Extraction Attack by Bayesian Active Watermarking},
author={Zhenyi Wang and Yihan Wu and Heng Huang},
booktitle={International Conference on Machine Learning},
year={2024}
}
```


## Questions?

For general questions, contact [Zhenyi Wang](wangzhenyineu@gmail.com)  </br>