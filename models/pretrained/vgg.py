import torch
import torchvision

def LMVGG(num_classes, use_gpu=True, model_name="vgg11"):
    if model_name == "vgg11":
        weights = torchvision.models.VGG11_Weights.DEFAULT
        model = torchvision.models.vgg11(weights=weights)
    elif model_name == "vgg11_bn":
        weights = torchvision.models.VGG11_BN_Weights.DEFAULT
        model = torchvision.models.vgg11_bn(weights=weights)
    elif model_name == "vgg13":
        weights = torchvision.models.VGG13_Weights.DEFAULT
        model = torchvision.models.vgg13(weights=weights)
    elif model_name == "vgg13_bn":
        weights = torchvision.models.VGG13_BN_Weights.DEFAULT
        model = torchvision.models.vgg13_bn(weights=weights)
    elif model_name == "vgg16":
        weights = torchvision.models.VGG16_Weights.DEFAULT
        model = torchvision.models.vgg16(weights=weights)
    elif model_name == "vgg16_bn":
        weights = torchvision.models.VGG16_BN_Weights.DEFAULT
        model = torchvision.models.vgg16_bn(weights=weights)
    elif model_name == "vgg19":
        weights = torchvision.models.VGG19_Weights.DEFAULT
        model = torchvision.models.vgg19(weights=weights)
    elif model_name == "vgg19_bn":
        weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        model = torchvision.models.vgg19_bn(weights=weights)
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
