import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.model import UNet
from src.dataloader import create_generators
from src.metrics import compute_metrics, get_metrics_name
from configs.utils import test_logger

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
                 hidden_channels=config.model.hidden_channels)

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
        crossentropy = torch.nn.CrossEntropyLoss()

        weigh = list(map(lambda x: 1 / x, CLASS_DISTRIBUTION))
        weigh = torch.tensor(weigh).to(device)
        crossentropy_weighted = torch.nn.CrossEntropyLoss(weight=weigh)

    else:
        raise 'please choose crossentropy loss'

    # Metrics
    acc = 'accuracy' in config.metrics
    iou_micro = 'iou_micro' in config.metrics
    iou_macro = 'iou_macro' in config.metrics
    iou_weighted = 'iou_weighted' in config.metrics
    metrics_name = get_metrics_name(acc=acc, iou_micro=iou_micro, iou_macro=iou_macro, iou_weighted=iou_weighted)

    # Evaluation
    test_loss = np.zeros(2, dtype=float)
    test_metrics = np.zeros(len(metrics_name), dtype=float)

    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc='evaluation'):
            image = image.to(device)
            y_true = target.to(device)

            y_pred = model(image)

            y_true = torch.movedim(y_true, 4, 1)

            test_loss += np.array([crossentropy(y_pred, y_true).item(),
                                   crossentropy_weighted(y_pred, y_true).item()])

            y_true = torch.movedim(y_true, 1, 4)
            y_pred = torch.movedim(y_pred, 1, 4)

            test_metrics += compute_metrics(y_true, y_pred, acc, iou_micro, iou_macro, iou_weighted, 1)

    test_loss = test_loss / len(test_loader)
    test_metrics = test_metrics / len(test_loader)

    name = np.concatenate((np.array(['crossentropy', 'crossentropy weighted'], ), metrics_name), axis=0)
    value = np.concatenate((test_loss, test_metrics), axis=0)

    for i in range(len(name)):
        print(str(name[i]) + ':', value[i])

    test_logger(logging_path, name, value)
        

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