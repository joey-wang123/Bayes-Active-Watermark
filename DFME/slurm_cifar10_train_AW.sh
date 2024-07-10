#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30gb
#SBATCH --job-name=seq-CIFAR10
#SBATCH --output=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/DFME/results2/newready-seq-CIFAR10-newAW-finetune-scale0.5-epoch210-1.out
#SBATCH --error=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/DFME/results2/newready-seq-CIFAR10-newAW-finetune-scale0.5-epoch210-1.err
#SBATCH --gres=gpu:rtxa5000:1


cd dfme;

source activate new
#srun python3 train_cifar10_AW.py --MAZE 0 --model resnet34_8x --dataset cifar10 --ckpt '/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/teacher-train/cache/models/resnet34_8x/new_CNN_encoding_data_dependent_random_weight_and_input_teacher_cifar10_scale_1.0/195.pth'   --device 0 --grad_m 1 --query_budget 20 --log_dir save_results/cifar10_MAZE0_rand_weight_teacher_resnet34_8x_student_resnet18_8x_budget_20  --lr_G 1e-4 --student_model resnet18_8x --loss l1;

srun python3 train_cifar10_AW.py --model resnet34_8x --dataset cifar10 --ckpt '/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/teacher-train/cache/models/resnet34_8x/newrandom_CNN_Active_Watermarking_teacher_cifar10/210.pth'   --device 0 --grad_m 1 --query_budget 20 --log_dir save_results/cifar10_MAZE0_rand_weight_teacher_resnet34_8x_student_resnet18_8x_budget_20  --lr_G 1e-4 --student_model resnet18_8x --loss l1;




