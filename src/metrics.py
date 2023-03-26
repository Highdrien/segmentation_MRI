import torch
import numpy as np
from sklearn.metrics import jaccard_score


def compute_metrics(y_true, y_pred, acc=True, iou_micro=True, iou_macro=True, iou_weighted=True, argmax_axis=1):
    metrics = []
    y_pred = torch.flatten(torch.argmax(y_pred, dim=argmax_axis)).cpu()
    y_true = torch.flatten(torch.argmax(y_true, dim=argmax_axis)).cpu()

    if acc:
        corrects = torch.eq(y_pred, y_true).float()
        acc = corrects.sum().item() / corrects.shape[0]
        metrics.append(acc)

    if iou_micro:
        metrics.append(jaccard_score(y_true, y_pred, average='micro'))

    if iou_macro:
        metrics.append(jaccard_score(y_true, y_pred, average='macro'))

    if iou_weighted:
        metrics.append(jaccard_score(y_true, y_pred, average='weighted'))

    return np.array(metrics)


def get_metrics_name(acc=True, iou_micro=True, iou_macro=True, iou_weighted=True):
    metrics_name = []
    if acc:
        metrics_name.append('accuracy')

    if iou_micro:
        metrics_name.append('iou_micro')

    if iou_macro:
        metrics_name.append('iou_macro')

    if iou_weighted:
        metrics_name.append('iou_weighted')

    return metrics_name


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
# print(get_metrics_name())
# print(compute_metrics(y_true, y_pred))