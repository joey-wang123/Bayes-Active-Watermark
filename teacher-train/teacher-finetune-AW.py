import os
from lenet import LeNet5
import torch
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from cifar10_models.noisy_resnet_cifar import *
from cifar10_models.vanilla_resnet_cifar import *
from network import resnet_8x
import argparse
from watermark_posterior import *
import random
from my_utils import *
from scipy.spatial.distance import pdist, squareform
from torch.autograd import grad
import time as t

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',help='input batch size for training (default: 256)')
parser.add_argument('--model', type=str, default='resnet34_8x', help='Target model name (default: resnet34_8x)')
parser.add_argument('--noisenet-max-eps', default=0.6, type=float)
parser.add_argument('--scale', default=0.5, type=float)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--num_classes', default=10, type=float)
parser.add_argument('--nz', type=int, default=256, help = "Size of random noise input to generator")
parser.add_argument('--approx_grad', type=int, default=1, help = 'Always set to 1')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')
args = parser.parse_args()


acc = 0
acc_best = 0

if args.dataset == 'MNIST':
    
    data_train = MNIST(args.data,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ]), download = True)
    data_test = MNIST(args.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]), download = True)

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=8)

    net = LeNet5().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
if args.dataset == 'cifar10':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR10(args.data,
                       transform=transform_train,
                       download=True)
    data_test = CIFAR10(args.data,
                      train=False,
                      transform=transform_test,
                      download=True)

    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=0)

    # net = resnet34(num_classes=10).cuda()
    # net = resnet_8x.ResNet34_8x(args, num_classes=10).cuda()
    # net = noise_resnet56(num_classes=10).cuda()
    # net = vanilla_resnet32(num_classes=10).cuda()


    net = get_classifier(args, args.model, pretrained=False, num_classes=args.num_classes).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

if args.dataset == 'cifar100':
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_train = CIFAR100(args.data,
                       transform=transform_train,
                       download=True)
    data_test = CIFAR100(args.data,
                      train=False,
                      transform=transform_test,
                      download=True)
                      
    data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    data_test_loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=0)

    net = resnet_8x.ResNet34_8x(args, num_classes=100).cuda()
    # net = noise_resnet56(num_classes=100).cuda()
    # net = vanilla_resnet32(num_classes=100).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)


use_cuda = True
device = torch.device("cuda:%d"%args.device if use_cuda else "cpu")
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# optimizer_AW = torch.optim.SGD(watermarknet.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)


channels = 3
args.G_activation = torch.tanh
generator = network.gan.GeneratorA(nz=args.nz, nc=channels, img_size=32, activation=args.G_activation)
generator = generator.to(device)


#calculate pairwise kernel distance
def kernal_dist(x, h=-1):

    x_numpy = x.cpu().data.numpy()
    init_dist = pdist(x_numpy)
    pairwise_dists = squareform(init_dist) 
    if h < 0:  # if h < 0, using median trick 
        h = np.median(pairwise_dists)
        h = 0.1*h ** 2 / np.log(x.shape[0] + 1)
    
    if x_numpy.shape[0]>1:
        kernal_xj_xi = torch.exp(- torch.tensor(pairwise_dists) ** 2 / h)
    else:
        kernal_xj_xi = torch.tensor([1])
    
    return kernal_xj_xi.to(x.device), h


def WGF_step(z_gen, tar_grad, lamb):
        
    kernal_xj_xi, h = kernal_dist(z_gen, h=-1)
    kernal_xj_xi, h = kernal_xj_xi.float(), h.astype(float)
    d_kernal_xi = torch.zeros(z_gen.size()).to(z_gen.device)

    F1_d_kernal_xi = torch.zeros(z_gen.size()).to(z_gen.device)
    F2_d_kernal_xi = torch.zeros(z_gen.size()).to(z_gen.device)
    F_delta = torch.zeros(z_gen.size()).to(z_gen.device)
    x = z_gen
    part_kernel = torch.sum(kernal_xj_xi, dim = 1).unsqueeze(1).to(x.device)
    for i_index in range(x.size()[0]):  
            quot_ele = torch.div(x[i_index] - x, part_kernel)
            F1_d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], quot_ele)* 2 / h
            F2_d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x)/(torch.sum(kernal_xj_xi[i_index])) * 2 / h      
            d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h
    
    for i_index in range(x.size()[0]):
            F_delta[i_index] =  tar_grad[i_index]  - F1_d_kernal_xi[i_index] - F2_d_kernal_xi[i_index]
    
    current_grad = (torch.matmul(kernal_xj_xi, tar_grad) + 0.2*d_kernal_xi)/x.size(0) + lamb*F_delta

    return current_grad



def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch, scale, generator):
    if args.dataset != 'MNIST':
        adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    lamb = 0.2


    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
    
            if args.batch_size > images.size(0):
                continue


            optimizer.zero_grad()
            net.train()
            ID_output, ID_AW = net(images, A_watermark=True)


            z = torch.randn((args.batch_size, args.nz)).to(device)
            #Get fake image from generator
            OODEgen = generator(z, pre_x=args.approx_grad) # pre_x returns the output of G before applying the activation
            OODEgen = torch.clamp(OODEgen, min=-1, max=1)

            net.eval()
            OOD_output, OOD_W = net(OODEgen, A_watermark=False)
            OODAW_output, OOD_AW = net(OODEgen, A_watermark=True)
            
            w_gen = torch.cat([ID_AW, OOD_AW])

            if i %2 ==0:

                loss = criterion(ID_output, labels) - args.alpha*criterion(OOD_output, OODAW_output)  
                loss_list.append(loss.data.item())
                batch_list.append(i+1)
    
                if i == 1:
                    print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
        
                loss.backward(retain_graph=True)

                w_gen = w_gen.view(w_gen.size(0), -1)
                tar_grad_ID = grad(-1.0*loss, ID_AW, torch.ones(loss.data.shape).cuda(),allow_unused=False, retain_graph=True)[0]
                tar_grad_ID = tar_grad_ID.view(tar_grad_ID.size(0), -1)


                tar_grad_OOD = grad(-1.0*loss, OOD_AW, torch.ones(loss.data.shape).cuda(),allow_unused=False, retain_graph=True)[0]
                tar_grad_OOD = tar_grad_OOD.view(tar_grad_OOD.size(0), -1)


                tar_grad = torch.cat([tar_grad_ID, tar_grad_OOD], dim = 0)
                current_grad = WGF_step(w_gen, tar_grad, lamb)
                w_gen.backward(-1.0*current_grad, retain_graph=True)
                optimizer.step()

            
                noise_factor = 1.0
                with torch.no_grad():
                    for param in generator.parameters():
                        param.add_(noise_factor * torch.randn(param.size()).to(device))

            else:    
                loss = criterion(ID_output, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
        

 
 
def test(scale):
    global acc, acc_best
    add_watermark = True
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()

            if args.batch_size > images.size(0):
                continue

            output, image_W = net(images, A_watermark=True)

            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
 
 
def train_and_test(epoch, scale, generator):
    print('epoch ', epoch)
    train(epoch, scale, generator)
    test(scale)
 
 
def main():

    PATH = args.output_dir + f'{args.model}/newrandom_CNN_Active_Watermarking_teacher_{args.dataset}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    print("load pre-trained model and test")
    print('load model path', args.ckpt)


    #load pre-trained model
    pretrained_state_dict =  torch.load(args.ckpt)

    # Create a new state_dict to hold the combined weights
    new_state_dict = net.state_dict()

    # Step 4: Transfer the pre-trained weights to the new model
    # Loop through the pre-trained model's state_dict
    for name, param in pretrained_state_dict.items():
        if name in new_state_dict:
            new_state_dict[name] = param

    # Load the new state_dict into the new model
    net.load_state_dict(new_state_dict)

    scale = args.scale
    test(scale)

    if args.dataset == 'MNIST':
        epoch = 10
    else:
        epoch = 200

    tuneepochs = 10
    for e in range(epoch, epoch+tuneepochs+1):
        print('fine-tuning')
        train_and_test(e, scale, generator)

    torch.save(net.state_dict(), PATH + f'{e}.pth')


 
if __name__ == '__main__':
    main()