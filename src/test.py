import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.model import UNet
from src.dataloader import create_generators
from src.metrics import compute_metrics
from configs.utils import test_logger
from src.loss import IoULoss, IoUClassesLoss

torch.manual_seed(0)
CLASS_DISTRIBUTION = [0.9621471811176255, 0.012111862189784502, 0.013016226246835367, 0.01272473044575458]


def evaluate(logging_path, config):
    
    # Construct Data loader

    dataset_test = create_generators(config)[2]

    test_loader = DataLoader(dataset_test, batch_size=config.test.batch_size)

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model
    model = UNet(input_channels=config.model.input_channels,
                 output_classes=config.model.output_channels,
                 hidden_channels=config.model.hidden_channels,
                 dropout_probability=config.model.dropout)

    model.to(device)

    # Load model's weight
    checkpoint_path = get_checkpoint_path(config, logging_path)
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    del checkpoint  # dereference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    # Loss
    if 'crossentropy' in config.model.loss.lower():
        if 'weigh' in config.model.loss.lower():
            weigh = list(map(lambda x: 1 / x, CLASS_DISTRIBUTION))
            weigh = torch.tensor(weigh).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weigh)
            print('loss:', 'cross entropy weighted')
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print('loss:', 'cross entropy')

    elif 'iou' in config.model.loss.lower():
        if 'classes' in config.model.loss.lower() or 'macro' in config.model.loss.lower():
            criterion = IoUClassesLoss(nb_classes=config.data.number_classes, smooth=0.001)
            print('loss:', 'iou classes')
        else:
            print('loss:', 'iou')
            criterion = IoULoss(smooth=0.001)

    else:
        raise 'please choose crossentropy loss or iou loss'

    # Metrics
    metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))

    # Evaluation
    test_loss = []
    test_metrics = np.zeros(len(metrics_name), dtype=float)

    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc='evaluation'):
            image = image.to(device)
            y_true = target.to(device)

            y_pred = model(image)

            y_true = torch.movedim(y_true, 4, 1)

            test_loss.append(criterion(y_pred, y_true).item())

            # y_true = torch.movedim(y_true, 1, 4)
            # y_pred = torch.movedim(y_pred, 1, 4)

            test_metrics += compute_metrics(config, y_true, y_pred, argmax_axis=1)

    if test_loss[-1] == 0:
        y_save('y_true.txt', y_true)
        y_save('y_pred.txt', y_pred)

    test_loss = sum(test_loss) / len(test_loader)
    test_metrics = test_metrics / len(test_loader)

    print('test loss:', test_loss)

    # name = np.concatenate((np.array(['crossentropy', 'crossentropy weighted'], ), metrics_name), axis=0)
    # value = np.concatenate((test_loss, test_metrics), axis=0)
    #
    # for i in range(len(name)):
    #     print(str(name[i]) + ':', value[i])

    test_logger(logging_path, metrics_name, test_metrics)
        

def get_checkpoint_path(config, path):
    pth_in_path = list(filter(lambda x: x[-3:] == 'pth', os.listdir(path)))

    if len(pth_in_path) == 1:
        return os.path.join(path, pth_in_path[0])

    if len(pth_in_path) == 0 and 'checkpoint_path' in os.listdir(path):
        model_path = os.path.join(path, 'checkpoint_path')

        if config.test.checkpoint in os.listdir(model_path):
            return os.path.join(model_path, config.test.checkpoint)

        elif config.test.checkpoint == 'last':
            pth_in_checkpoint = list(filter(lambda x: x[-3:] == 'pth', os.listdir(model_path)))
            model_name = 'model' + str(len(pth_in_checkpoint) - 1) + 'pth'
            return os.path.join(model_path, model_name)

        elif 'model' + config.test.checkpoint + 'pth' in os.listdir(model_path):
            return os.path.join(model_path, 'model' + config.test.checkpoint + 'pth')

    elif config.test.checkpoint == 'last':
        return os.path.join(path, pth_in_path[-1])

    elif 'model' + config.test.checkpoint + 'pth' in os.listdir(path):
        return os.path.join(path, 'model' + config.test.checkpoint + 'pth')

    raise 'The model weights could not be found'


def y_save(path, y):
    with open(path, 'w') as file:
        file.write(str(y.shape))
        for x in tqdm(torch.flatten(y), desc='save_y'):
            file.write(str(x.item()) + '\n')