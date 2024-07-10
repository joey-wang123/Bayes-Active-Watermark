#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30gb
#SBATCH --job-name=seq-CIFAR100
#SBATCH --output=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/DFME/results2/random-newresnet-watermark-seq-CIFAR100-newAW-finetune-scale0.5-epoch210-1.out
#SBATCH --error=/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/DFME/results2/random-newresnet-watermark-seq-CIFAR100-newAW-finetune-scale0.5-epoch210-1.err
#SBATCH --gres=gpu:rtxa6000:1


cd dfme;

source activate new
python3 train_cifar100_AW.py --model resnet34_8x --dataset cifar100 --ckpt '/fs/nexus-scratch/joeywang/code/DFMEdefense-newAW/teacher-train/cache/models/resnet34_8x/newrandom_CNN_Active_Watermarking_teacher_cifar100/210.pth'   --device 0 --grad_m 1 --query_budget 200 --log_dir save_results/cifar100_cleandata_DFME_rand_teacher_resnet34_8X_student_resnet18_8x_2 --lr_G 1e-4 --student_model  resnet18_8x --loss l1;


