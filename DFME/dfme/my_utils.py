from cifar10_models import *
from approximate_gradients import *
import network

def get_classifier(args, classifier, pretrained=True, num_classes=10):
    if classifier.lower()=='lenet5':
        return network.lenet.LeNet5()

    if classifier.lower()=='lenet5half':
        return network.lenet.LeNet5Half()

    if classifier == "wrn-28-10":
        net =  wrn(
                    num_classes=num_classes,
                    depth=28,
                    widen_factor=10,
                    dropRate=0.3
                )
        if pretrained:
            state_dict = torch.load("cifar100_models/state_dicts/model_best.pt", map_location=device)["state_dict"]
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

        return net
    elif 'wrn' in classifier and 'kt' not in classifier:
        depth = int(classifier.split("-")[1])
        width = int(classifier.split("-")[2])

        net =  wrn(
                    num_classes=num_classes,
                    depth=depth,
                    widen_factor=width
                )
        if pretrained:
            raise ValueError("Cannot be pretrained")
        return net
    elif classifier == "kt-wrn-28-2":
        print('loading pre-trained classifier kt-wrn-28-2')
        net = WideResNetKT(args, depth=28, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        if pretrained:
            state_dict = torch.load(args.ckpt)
            net.load_state_dict(state_dict)
        return net
    elif classifier == "resnet34_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    elif classifier == "resnet18_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet18_8x from here")
        return network.resnet_8x.ResNet18_8x(num_classes=num_classes)

    else:
        raise NameError('Please enter a valid classifier')


def measure_true_grad_norm(args, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(args, args.teacher, args.student, x, pre_x=True, device=args.device)
    if args.dataset == 'MNIST':
        size = 1024
    else:
        size = 3072
    true_grad = true_grad.view(-1, size)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad

classifiers = [
    'lenet5',
    'lenet5half',
    'lenet5fifth',
    "resnet34_8x", # Default DFAD
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet161",
    "densenet169",
    "mobilenet_v2",
    "googlenet",
    "inception_v3",
    "wrn-28-10",
    "resnet18_8x",
    "kt-wrn-28-2",
]