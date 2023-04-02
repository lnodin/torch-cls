import torchvision.transforms as transforms

from typing import List


def lm_transform(lst_trans_operations: List = [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]):
    return transforms.Compose(
        lst_trans_operations
    )
