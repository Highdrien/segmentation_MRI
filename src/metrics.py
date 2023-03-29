import torch
import numpy as np
# import random
from sklearn.metrics import jaccard_score

# random.seed(0)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

CLASS_DISTRIBUTION = [0.9621471811176255, 0.012111862189784502, 0.013016226246835367, 0.01272473044575458]
WEIGH = torch.tensor(list(map(lambda x: 1 / x, CLASS_DISTRIBUTION))).to(device)


def compute_metrics(config, y_true, y_pred, argmax_axis=1):

    crossentropy = []
    if config.metrics.crossentropy:
        criterion = torch.nn.CrossEntropyLoss()
        crossentropy.append(criterion(y_pred, y_true).item())

    if config.metrics.crossentropy_weighted:
        criterion = torch.nn.CrossEntropyLoss(weight=WEIGH)
        crossentropy.append(criterion(y_pred, y_true).item())

    metrics = []
    y_true = torch.movedim(y_true, 1, 4)
    y_pred = torch.movedim(y_pred, 1, 4)
    y_pred = torch.flatten(torch.argmax(y_pred, dim=argmax_axis)).cpu()
    y_true = torch.flatten(torch.argmax(y_true, dim=argmax_axis)).cpu()

    if config.metrics.accuracy:
        corrects = torch.eq(y_pred, y_true).float()
        acc = corrects.sum().item() / corrects.shape[0]
        metrics.append(acc)

    if config.metrics.iou_micro:
        metrics.append(jaccard_score(y_true, y_pred, average='micro'))

    if config.metrics.iou_macro:
        metrics.append(jaccard_score(y_true, y_pred, average='macro'))

    if config.metrics.iou_weighted:
        metrics.append(jaccard_score(y_true, y_pred, average='weighted'))

    return np.array(metrics + crossentropy)


# def create_label(shape):
#     y = torch.zeros(shape, dtype=torch.float)
#     for i in range(shape[1]):
#         for j in range(shape[3]):
#             for k in range(shape[4]):
#                 idx = random.randint(0, shape[2] - 1)
#                 y[0, i, idx, j, k] = 1
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
# if __name__ == '__main__':
#     shape = [1, 4, 256, 256, 10]
#     y_true = create_label(shape)
#     y_pred = create_pred(shape)
#
#     print('iou', 1 - IoU(y_true, y_pred))
#     print('iou classes', IoU_classes(y_true, y_pred, 4))
