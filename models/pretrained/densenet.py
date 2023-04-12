import torch
import torchvision

def LMDenseNet(model_name, num_classes, use_gpu=True):
    if model_name == 'densenet121':
        weights = torchvision.models.DenseNet121_Weights.DEFAULT
        model = torchvision.models.densenet121(weights=weights)
    elif model_name == 'densenet161':
        weights = torchvision.models.DenseNet161_Weights.DEFAULT
        model = torchvision.models.densenet161(weights=weights)
    elif model_name == 'densenet169':
        weights = torchvision.models.DenseNet169_Weights.DEFAULT
        model = torchvision.models.densenet169(weights=weights)
    else:
        weights = torchvision.models.DenseNet201_Weights.DEFAULT
        model = torchvision.models.densenet201(weights=weights)

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