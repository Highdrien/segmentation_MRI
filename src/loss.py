import torch
from torch import nn


def iou(y_true, y_pred, smooth=0.001):
    # flatten label and prediction tensors
    inter = torch.sum(y_true * y_pred)
    union = torch.sum(y_true + y_pred) - inter

    return (inter + smooth) / (union + smooth)


# PyTorch
class IoULoss(nn.Module):
    def __init__(self, smooth=0.001):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        return 1 - iou(y_true, y_pred, self.smooth)


class IoUClassesLoss(nn.Module):
    def __init__(self, nb_classes, smooth=0.001):
        super(IoUClassesLoss, self).__init__()
        self.nb_classes = nb_classes
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        iou_classes = [iou(y_true[:, :, :, i], y_pred[:, :, :, i], self.smooth) for i in range(self.nb_classes)]
        return 1 - sum(iou_classes) / self.nb_classes
