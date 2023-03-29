import torch
from tqdm import tqdm


def IoU(y_true, y_pred, smooth=0.001):
    # flatten label and prediction tensors
    inter = torch.sum(y_true * y_pred)
    union = torch.sum(y_true + y_pred) - inter

    return (inter + smooth) / (union + smooth)


def IoU_classes(y_true, y_pred, nb_classes, smooth=0.001):
    iou_classes = [IoU(y_true[:, i, :, :], y_pred[:, i, :, :], smooth) for i in range(nb_classes)]
    print(iou_classes)
    return 1 - sum(iou_classes) / nb_classes


def load_y(path):
    shape = (1, 4, 176, 224, 8)
    y = torch.zeros(shape)
    y = y.type(torch.float)
    f = open(path, "r")
    f.readline()
    for i0 in range(shape[0]):
        for i1 in tqdm(range(shape[1])):
            for i2 in range(shape[2]):
                for i3 in range(shape[3]):
                    for i4 in range(shape[4]):
                        y[i0, i1, i2, i3, i4] = float(f.readline())
    return y


if __name__ == '__main__':
    y_true = load_y('y_true.txt')
    y_pred = load_y('y_pred.txt')

    print('iou:', 1 - IoU(y_true, y_pred))
    print('iou classes:', IoU_classes(y_true, y_pred, nb_classes=4))