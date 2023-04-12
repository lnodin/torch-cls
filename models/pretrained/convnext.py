import torch
import torchvision

def LMConvNext(model_name, num_classes, use_gpu=True):
    if model_name == 'convnext_tiny':
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        model = torchvision.models.convnext_tiny(weights=weights)
    elif model_name == 'convnext_small':
        weights = torchvision.models.ConvNeXt_Small_Weights.DEFAULT
        model = torchvision.models.convnext_small(weights=weights)
    elif model_name == 'convnext_base':
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
        model = torchvision.models.convnext_base(weights=weights)

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