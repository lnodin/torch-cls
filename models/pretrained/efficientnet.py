import torch
import torchvision


def LMEfficientNet(model_name, num_classes, use_gpu=True):
    if model_name == 'efficientnet_b0':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif model_name == 'efficientnet_b1':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
        model = torchvision.models.efficientnet_b1(weights=weights)
    elif model_name == 'efficientnet_b2':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
    elif model_name == 'efficientnet_b3':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
        model = torchvision.models.efficientnet_b3(weights=weights)
    elif model_name == 'efficientnet_b4':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        model = torchvision.models.efficientnet_b4(weights=weights)
    elif model_name == 'efficientnet_b5':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
        model = torchvision.models.efficientnet_b5(weights=weights)
    elif model_name == 'efficientnet_b6':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT
        model = torchvision.models.efficientnet_b6(weights=weights)
    elif model_name == 'efficientnet_b7':
        # .DEFAULT = best available weights
        weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
        model = torchvision.models.efficientnet_b7(weights=weights)
    if use_gpu:
        model = model.cuda()

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=num_classes,  # same number of output units as our number of classes
                        bias=True)).cuda()

    return model
