#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30gb
#SBATCH --job-name=seq-CIFAR10
#SBATCH --output=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/DFME-noise/results/seq-CIFAR10-newAW-finetune-standardDFME-batchsize64-3.out
#SBATCH --error=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/DFME-noise/results/seq-CIFAR10-newAW-finetune-standardDFME-batchsize64-3.err
#SBATCH --gres=gpu:rtxa5000:1


cd dfme;

source activate new

srun python3 train.py --MAZE 0 --model resnet34_8x --dataset cifar10 --ckpt '/fs/nexus-scratch/joeywang/code/DFMEdefenseAW2/teacher-train/cache/models/resnet34_8x/CNN_standard_teacher_cifar10/195.pth'  --device 0 --grad_m 1 --query_budget 20 --log_dir save_results/cifar10_MAZE0_rand_weight_teacher_resnet34_8x_student_resnet18_8x_budget_20  --lr_G 1e-4 --student_model resnet18_8x --loss l1;




