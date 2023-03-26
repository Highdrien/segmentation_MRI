import torch
import numpy as np
# import random
from sklearn.metrics import jaccard_score

# random.seed(0)
CLASS_DISTRIBUTION = [0.9621471811176255, 0.012111862189784502, 0.013016226246835367, 0.01272473044575458]
WEIGH = torch.tensor(list(map(lambda x: 1 / x, CLASS_DISTRIBUTION)))


def compute_metrics(config, y_true, y_pred, argmax_axis=1):

    crossentropy = []
    if config.metrics.crossentropy:
        criterion = torch.nn.CrossEntropyLoss()
        crossentropy.append(criterion(y_pred, y_true).item())

    if config.metrics.crossentropy_weighted:
        criterion = torch.nn.CrossEntropyLoss(weight=WEIGH)
        crossentropy.append(criterion(y_pred, y_true).item())

    metrics = []
    print(f'{y_pred.shape = }')
    print(f'{y_true.shape = }')
    y_true = torch.movedim(y_true, 1, 4)
    y_pred = torch.movedim(y_pred, 1, 4)
    print(f'{y_pred.shape = }')
    print(f'{y_true.shape = }')
    y_pred = torch.flatten(torch.argmax(y_pred, dim=argmax_axis)).cpu()
    y_true = torch.flatten(torch.argmax(y_true, dim=argmax_axis)).cpu()
    print(f'{y_pred.shape = }')
    print(f'{y_true.shape = }')

    if config.metrics.accuracy:
        corrects = torch.eq(y_pred, y_true).float()
        acc = corrects.sum().item() / corrects.shape[0]
        print(acc)
        metrics.append(acc)

    if config.metrics.iou_micro:
        metrics.append(jaccard_score(y_true, y_pred, average='micro'))

    if config.metrics.iou_macro:
        metrics.append(jaccard_score(y_true, y_pred, average='macro'))

    if config.metrics.iou_weighted:
        metrics.append(jaccard_score(y_true, y_pred, average='weighted'))

    return np.array(metrics + crossentropy)


def get_metrics_name(config):
    metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))
    return metrics_name


# def IoU(y_true, y_pred, smooth=0.001):
#     # flatten label and prediction tensors
#     inter = torch.sum(y_true * y_pred)
#     union = torch.sum(y_true + y_pred) - inter
#
#     return 1 - (inter + smooth) / (union + smooth)
#
#
# def IoU_classes(y_true, y_pred, nb_classes, smooth=0.001):
#     iou_classes = torch.tensor([IoU(y_true[:, :, :, i], y_pred[:, :, :, i], smooth) for i in range(nb_classes)])
#
#     return iou_classes.mean()
#
#
# def create_label(shape):
#     y = torch.zeros(shape, dtype=torch.float)
#     for i in range(shape[1]):
#         for j in range(shape[2]):
#             for k in range(shape[3]):
#                 idx = random.randint(0, shape[4] - 1)
#                 y[0, i, j, k, idx] = 1
#     return y
#
#
# def create_pred(shape):
#     y = torch.zeros(shape, dtype=torch.float)
#     for i in range(shape[1]):
#         for j in range(shape[2]):
#             for k in range(shape[3]):
#                 row = torch.tensor([random.random() for _ in range(shape[4])])
#                 y[0, i, j, k, :] = row / torch.sum(row)
#     return y
#
#
# shape = [1, 4, 10, 10, 6]
# y_true = create_label(shape)
# y_pred = create_pred(shape)
#
# print(compute_metrics(y_true, y_pred, acc=False, iou_micro=True, iou_macro=True, iou_weighted=False))
#
# print('iou', IoU(y_true, y_pred))
# print('iou classes', IoU_classes(y_true, y_pred, 4))
