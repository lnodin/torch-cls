import torch
import torchvision

def LMVisionTransformer(num_classes, use_gpu=True, model_name="vit_b_16"):
    if model_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
    elif model_name == "vit_b_32":
        weights = torchvision.models.ViT_B_32_Weights.DEFAULT
        model = torchvision.models.vit_b_32(weights=weights)
    elif model_name == "vit_l_16":
        weights = torchvision.models.ViT_L_16_Weights.DEFAULT
        model = torchvision.models.vit_l_16(weights=weights)
    elif model_name == "vit_l_32":
        weights = torchvision.models.ViT_L_32_Weights.DEFAULT
        model = torchvision.models.vit_l_32(weights=weights)
    elif model_name == "vit_h_14":
        weights = torchvision.models.ViT_H_14_Weights.DEFAULT
        model = torchvision.models.vit_h_14(weights=weights)
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