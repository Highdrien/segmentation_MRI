import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.model import UNet
from src.metrics import compute_metrics
from src.dataloader import create_generators
from src.loss import IoULoss, IoUClassesLoss
from configs.utils import train_logger, train_step_logger


torch.manual_seed(0)
CLASS_DISTRIBUTION = [0.9621471811176255, 0.012111862189784502, 0.013016226246835367, 0.01272473044575458]


def train(config):
    dataset_train, dataset_val, _ = create_generators(config)
    train_loader = DataLoader(dataset_train, batch_size=config.train.batch_size)
    val_loader = DataLoader(dataset_val, batch_size=config.val.batch_size)

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Model
    model = UNet(input_channels=config.model.input_channels,
                 output_classes=config.model.output_channels,
                 hidden_channels=config.model.hidden_channels,
                 dropout_probability=config.model.dropout)

    model.to(device)
    print('number of parameters:', get_n_params(model))

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

    # Optimizer
    if config.model.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    else:
        raise 'the Adam optimizer is the only one to be implemented'

    # Metrics
    metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))
    logging_path = train_logger(config, metrics_name)

    best_epoch, best_val_loss = 0, 10e6

    ###############################################################
    # Start Training                                              #
    ###############################################################
    model.train()

    for epoch in range(1, config.train.epochs + 1):
        print('epoch:' + str(epoch))
        train_loss = []
        train_metrics = np.zeros(len(metrics_name), dtype=float)
        virtual_batch_counter = 0

        train_range = tqdm(train_loader)
        for (image, target) in train_range:

            image = image.to(device)
            y_true = target.to(device)

            y_pred = model(image)

            y_true = torch.movedim(y_true, 4, 1)

            loss = criterion(y_pred, y_true)
            loss.backward()

            train_loss.append(loss.item())

            train_metrics += compute_metrics(config, y_true, y_pred, argmax_axis=-1)

            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, np.mean(train_loss)))
            train_range.refresh()

            if virtual_batch_counter % config.train.virtual_batch_size == 0 or virtual_batch_counter + 1 == len(
                    train_loader):
                optimizer.step()
                optimizer.zero_grad()

            virtual_batch_counter += 1

        train_metrics = train_metrics / len(train_loader)

        ###############################################################
        # Start Evaluation                                            #
        ###############################################################

        model.eval()
        val_loss = []
        val_metrics = np.zeros(len(metrics_name), dtype=float)

        with torch.no_grad():
            for (image, target) in tqdm(val_loader, desc='validation'):
                image = image.to(device)
                y_true = target.to(device)

                y_pred = model(image)

                y_true = torch.movedim(y_true, 4, 1)

                loss = criterion(y_pred, y_true)
                val_loss.append(loss.item())

                val_metrics += compute_metrics(config, y_true, y_pred, argmax_axis=-1)

        val_loss = np.mean(val_loss)
        val_metrics = val_metrics / len(val_loader)

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################

        train_step_logger(logging_path, epoch, np.mean(train_loss), val_loss, train_metrics, val_metrics)

        if config.train.save_checkpoint.lower() == 'all':
            checkpoint_path = os.path.join(logging_path, 'checkpoint_path')
            checkpoint_name = 'model' + str(epoch) + 'pth'
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))

        elif config.train.save_checkpoint.lower() == 'best':
            if val_loss < best_val_loss:
                print('saving checkpoints')
                best_epoch, best_val_loss = epoch, val_loss
                torch.save(model.state_dict(), os.path.join(logging_path, 'model.pth'))

    if config.train.save_checkpoint.lower() == 'best':
        old_name = os.path.join(logging_path, 'model.pth')
        new_name = os.path.join(logging_path, 'model' + str(best_epoch) + '.pth')
        os.rename(old_name, new_name)

    elif config.train.save_checkpoint == 'last':
        torch.save(model.state_dict(), os.path.join(logging_path, 'model.pth'))

    if config.train.save_learning_curves:
        save_learning_curves(logging_path)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def save_learning_curves(path):
    save_path = os.path.join(path, 'learning_curves')
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        name = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()

    os.makedirs(save_path, exist_ok=True)
    for i in range(1, len(name), 2):
        epoch = result[:, 0]
        train_metric = result[:, i]
        val_metric = result[:, i + 1]
        plt.title(name[i])
        plt.plot(epoch, train_metric)
        plt.plot(epoch, val_metric)
        plt.xlabel('epoch')
        plt.ylabel(name[i])
        plt.legend(['train', 'val'])
        plt.grid()
        plt.savefig(os.path.join(save_path, name[i] + '.png'))
        plt.close()
