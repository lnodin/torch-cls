import torch
import torchvision

def LMSqueezeNet(num_classes, use_gpu=True, model_name="wide_resnet50_2"):
    if model_name == "squeezenet1_0":
        weights = torchvision.models.SqueezeNet1_0_Weights.DEFAULT
        model = torchvision.models.squeezenet1_0(weights=weights)
    else:
        weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT
        model = torchvision.models.squeezenet1_1(weights=weights)
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