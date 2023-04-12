import torch
import torchvision

def LMResNext(num_classes, use_gpu=True, model_name="resnext50_32x4d"):
    if model_name == "resnext50_32x4d":
        weights = torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
        model = torchvision.models.resnext50_32x4d(weights=weights)
    elif model_name == "resnext101_32x8d":
        weights = torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT
        model = torchvision.models.resnext101_32x8d(weights=weights)
    elif model_name == "resnext101_64x4d":
        weights = torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT
        model = torchvision.models.resnext101_64x4d(weights=weights)
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
