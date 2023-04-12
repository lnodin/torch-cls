import torch
import torchvision

def LMSwinTransformer(num_classes, use_gpu=True, model_name="swin_t"):
    if model_name == "swin_t":
        weights = torchvision.models.Swin_T_Weights.DEFAULT
        model = torchvision.models.swin_t(weights=weights)
    elif model_name == "swin_s":
        weights = torchvision.models.Swin_S_Weights.DEFAULT
        model = torchvision.models.swin_s(weights=weights)
    elif model_name == "swin_b":
        weights = torchvision.models.Swin_B_Weights.DEFAULT
        model = torchvision.models.swin_b(weights=weights)
    elif model_name == "swin_v2_t":
        weights = torchvision.models.Swin_V2_T_Weights.DEFAULT
        model = torchvision.models.swin_v2_t(weights=weights)
    elif model_name == "swin_v2_s":
        weights = torchvision.models.Swin_V2_S_Weights.DEFAULT
        model = torchvision.models.swin_v2_s(weights=weights)
    elif model_name == "swin_v2_b":
        weights = torchvision.models.Swin_V2_B_Weights.DEFAULT
        model = torchvision.models.swin_v2_b(weights=weights)
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
