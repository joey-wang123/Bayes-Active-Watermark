#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30gb
#SBATCH --job-name=seq-CIFAR10
#SBATCH --output=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/teacher-train/results2/ready-teacher-newrandom-seq-CIFAR100-finetune-AW-teacher-scale0.5-perturb-2.out
#SBATCH --error=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/teacher-train/results2/ready-teacher-newrandom-seq-CIFAR100-finetune-AW-teacher-scale0.5-perturb-2.err
#SBATCH --gres=gpu:rtxa5000:1


cd dfme;

source activate new

# srun python teacher-train-AW.py --dataset cifar10 --model resnet34_8x --scale 1.0 --alpha 0.00001

#srun python teacher-finetune-AW.py --dataset cifar10 --model resnet34_8x --scale 0.5 --alpha 0.00001 --ckpt '/fs/nexus-scratch/joeywang/code/DFMEdefenseAW2/teacher-train/cache/models/resnet34_8x/CNN_standard_teacher_cifar10/195.pth'


srun python teacher-finetune-AW.py --dataset cifar100 --model resnet34_8x --scale 0.5 --alpha 0.00001 --ckpt 'models/CIFAR100.pth'