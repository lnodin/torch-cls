import torch
import numpy as np

import logging

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class Tester(object):
    def __init__(self,
                 model,
                 classes,
                 test_dataloader=None,
                 use_gpu=True
                 ) -> None:
        self.model = model
        self.test_dataloader = test_dataloader
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
            self.model = self.model.cuda()
        self.classes = classes
        self.y_actual = []
        self.y_pred = []

    def run(self):
        correct = 0
        total = 0
        for images, labels in self.test_dataloader:
            if self.use_gpu:
                images = images.to(self.device)
                labels = labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            self.y_actual.append(labels.cpu().numpy())
            self.y_pred.append(predicted.cpu().numpy())
        logging.info('Accuracy of the network on test images: {:.5f} '.format(correct / total))
        print(classification_report(self.y_actual, self.y_pred, target_names=self.classes))
        
