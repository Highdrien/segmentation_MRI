import torch
from tqdm import tqdm

from src.metrics import IoU, IoU_classes


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