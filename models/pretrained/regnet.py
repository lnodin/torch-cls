import torch
import torchvision

# need rename something
def LMRegNet(num_classes, use_gpu=True, model_name="regnet_y_400mf"):
    if model_name == "regnet_y_400mf":
        weights = torchvision.models.RegNet_Y_400MF_Weights.DEFAULT
        model = torchvision.models.regnet_y_400mf(weights=weights)
    elif model_name == "regnet_y_800mf":
        weights = torchvision.models.RegNet_Y_800MF_Weights.DEFAULT
        model = torchvision.models.regnet_y_800mf(weights=weights)
    elif model_name == "regnet_y_1_6gf":
        weights = torchvision.models.RegNet_Y_1_6GF_Weights.DEFAULT
        model = torchvision.models.vgg13(weights=weights)
    elif model_name == "regnet_y_3_2gf":
        weights = torchvision.models.VGG13_BN_Weights.DEFAULT
        model = torchvision.models.vgg13_bn(weights=weights)
    elif model_name == "regnet_y_8gf":
        weights = torchvision.models.VGG16_Weights.DEFAULT
        model = torchvision.models.vgg16(weights=weights)
    elif model_name == "regnet_y_16gf":
        weights = torchvision.models.VGG16_BN_Weights.DEFAULT
        model = torchvision.models.vgg16_bn(weights=weights)
    elif model_name == "regnet_y_32gf":
        weights = torchvision.models.VGG19_Weights.DEFAULT
        model = torchvision.models.vgg19(weights=weights)
    elif model_name == "regnet_y_128gf":
        weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        model = torchvision.models.vgg19_bn(weights=weights)
    elif model_name == "regnet_x_400mf":
        weights = torchvision.models.VGG16_BN_Weights.DEFAULT
        model = torchvision.models.vgg16_bn(weights=weights)
    elif model_name == "regnet_x_800mf":
        weights = torchvision.models.VGG19_Weights.DEFAULT
        model = torchvision.models.vgg19(weights=weights)
    elif model_name == "regnet_x_1_6gf":
        weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        model = torchvision.models.vgg19_bn(weights=weights)
    elif model_name == "regnet_x_3_2gf":
        weights = torchvision.models.VGG19_BN_Weights.DEFAULT
        model = torchvision.models.vgg19_bn(weights=weights)
    elif model_name == "regnet_x_8gf":
        weights = torchvision.models.VGG16_BN_Weights.DEFAULT
        model = torchvision.models.vgg16_bn(weights=weights)
    elif model_name == "regnet_x_16gf":
        weights = torchvision.models.VGG19_Weights.DEFAULT
        model = torchvision.models.vgg19(weights=weights)
    elif model_name == "regnet_x_32gf":
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
