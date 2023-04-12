import torch
import torchvision

def LMShuffleNetV2(num_classes, use_gpu=True, model_name="shufflenet_v2_x0_5"):
    if model_name == "shufflenet_v2_x0_5":
        weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        model = torchvision.models.shufflenet_v2_x0_5(weights=weights)
    elif model_name == "shufflenet_v2_x1_0":
        weights = torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        model = torchvision.models.shufflenet_v2_x1_0(weights=weights)
    elif model_name == "shufflenet_v2_x1_5":
        weights = torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT
        model = torchvision.models.shufflenet_v2_x1_5(weights=weights)
    elif model_name == "shufflenet_v2_x2_0":
        weights = torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        model = torchvision.models.shufflenet_v2_x2_0(weights=weights)
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
