import os
import numpy as np
import matplotlib.pyplot as plt


def save_learning_curves(path):
    result, names = get_result(path)
    save_path = os.path.join(path, 'learning_curves')
    os.makedirs(save_path, exist_ok=True)
    loss_index, acc_index, iou_index, entropy_index = make_groups(names)
    plot_curves(result, names, save_path, names[1], loss_index)
    plot_curves(result, names, save_path, 'accuracy', acc_index)
    plot_curves(result, names, save_path, 'IoU', iou_index)
    plot_curves(result, names, save_path, 'CrossEntropy', entropy_index)


def get_result(path):
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        names = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()
    return result, names


def make_groups(name):
    index = list(range(len(name)))
    loss_index = list(filter(lambda x: 'loss' in name[x], index))
    acc_index = list(filter(lambda x: 'acc' in name[x] and x not in loss_index, index))
    iou_index = list(filter(lambda x: 'iou' in name[x] and x not in loss_index, index))
    entropy_index = list(filter(lambda x: 'entropy' in name[x] and x not in loss_index, index))
    return loss_index, acc_index, iou_index, entropy_index


def plot_curves(result, names, save_path, name, indexes):
    if len(indexes) == 0:
        return None
    epochs = result[:, 0]
    legend = []
    for i in indexes:
        metric = result[:, i]
        legend.append(names[i])
        plt.plot(epochs, metric)

    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.legend(legend)
    plt.grid()
    plt.savefig(os.path.join(save_path, name + '.png'))
    plt.close()