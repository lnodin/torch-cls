import torch
import torchvision
import numpy as np

# from models import ConvNet
from loss_funcs import CrossEntropyLoss
from trainers import Trainer, MPTrainer
from testers import Tester
from loggers import set_logger

from data_loaders.lm_dataloader import data_loader

set_logger(data_name='mnist', save_path='./loggers/log')

data_dir = 'data'

train_loader, valid_loader, test_loader, classes = data_loader(data_dir=data_dir,)

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# model = ConvNet(num_classes=len(classes)).cuda()
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_b0(weights=weights).cuda()

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=len(classes), # same number of output units as our number of classes
                    bias=True)).cuda()

print("Total number of parameters =", np.sum(
    [np.prod(parameter.shape) for parameter in model.parameters()]))

loss_func = CrossEntropyLoss()

trainer = MPTrainer(model, train_dataloader=train_loader, valid_dataloader=valid_loader,
                    train_epochs=3, valid_epochs=2, learning_rate=0.001, loss_func=loss_func, optimization_method='adam')

model, losses, accuracies = trainer.run()

# trainer.save_model('saved_models/vgg11_mnist.model')

# model_loaded = trainer.load_model('saved_models/vgg11_mnist.model')

tester = Tester(model=model, classes=classes,
                test_dataloader=test_loader, use_gpu=True)

tester.run()