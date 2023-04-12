import torch
import torchvision

def LMEfficientNet(model_name, num_classes, use_gpu=True):
    if model_name == 'efficientnet_v2_s':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_s(weights=weights)
    elif model_name == 'efficientnet_v2_m':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_m(weights=weights)
    else:
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_l(weights=weights)

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