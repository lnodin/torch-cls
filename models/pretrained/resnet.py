import torch
import torchvision

def LMResNet(num_classes, use_gpu=True, model_name="resnet18"):
    if model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT
        model = torchvision.models.resnet34(weights=weights)
    elif model_name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
    elif model_name == "resnet101":
        weights = torchvision.models.ResNet101_Weights.DEFAULT
        model = torchvision.models.resnet101(weights=weights)
    elif model_name == "resnet152":
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        model = torchvision.models.resnet152(weights=weights)
    else:
        raise NameError

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