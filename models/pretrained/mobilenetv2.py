import torch
import torchvision

def LMMobileNetV2(num_classes, use_gpu=True, model_name="alexnet"):
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights)
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