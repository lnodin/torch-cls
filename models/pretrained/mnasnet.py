import torch
import torchvision

def LMMNASNet(model_name, num_classes, use_gpu=True):
    if model_name == 'mnasnet0_5':
        weights = torchvision.models.MNASNet0_5_Weights.DEFAULT
        model = torchvision.models.mnasnet0_5(weights=weights)
    elif model_name == 'mnasnet0_75':
        weights = torchvision.models.MNASNet0_75_Weights.DEFAULT
        model = torchvision.models.mnasnet0_75(weights=weights)
    elif model_name == 'mnasnet1_0':
        weights = torchvision.models.MNASNet1_0_Weights.DEFAULT
        model = torchvision.models.mnasnet1_0(weights=weights)
    else:
        weights = torchvision.models.MNASNet1_3_Weights.DEFAULT
        model = torchvision.models.mnasnet1_3(weights=weights)

    if use_gpu:
        model = model.cuda()
        model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=num_classes,  # same number of output units as our number of classes
                        bias=True)).cuda()
        return model
    else:
        model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=num_classes,  # same number of output units as our number of classes
                        bias=True))
        return model