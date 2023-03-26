import os
from datetime import datetime
from easydict import EasyDict as edict


def number_folder(path, name):
    """
    finds a declination of a folder name so that the name is not already taken
    """
    elements = os.listdir(path)
    last_index = -1
    for i in range(len(elements)):
        folder_name = name + str(i)
        if folder_name in elements:
            last_index = i
    return name + str(last_index + 1)


def train_logger(config, metrics_name):
    """
    creates a logs folder where we can find the config in confing.yaml and
    the values of the loss and metrics according to the epochs in train_log.csv
    """
    path = config.train.logs_path
    folder_name = number_folder(path, 'experiment_')
    path = os.path.join(path, folder_name)
    os.mkdir(path)
    print(f'{path = }')

    # create train_log.csv where save the metrics
    with open(os.path.join(path, 'train_log.csv'), 'w') as f:
        first_line = 'step,' + config.model.loss + ',val ' + config.model.loss
        for metric in metrics_name:
            first_line += ',' + config.metrics[metric]
            first_line += ',val ' + config.metrics[metric]
        f.write(first_line + '\n')
    f.close()

    # copy the config
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        f.write("config_metadata: 'Saving time : " + date_time + "'\n")
        for line in config_to_yaml(config):
            f.write(line + '\n')
    f.close()

    return path


def config_to_yaml(config, space=''):
    """
    transforms a dictionary (config) into a yaml line sequence
    """
    config_str = []
    for key, value in config.items():
        if type(value) == edict:
            config_str.append('')
            config_str.append('# ' + key + ' options')
            config_str.append(key + ':')
            config_str += config_to_yaml(value, space='  ')
        elif type(value) == str:
            config_str.append(space + key + ": '" + str(value) + "'")
        elif value is None:
            config_str.append(space + key + ": null")
        elif type(value) == bool:
            config_str.append(space + key + ": " + str(value).lower())
        else:
            config_str.append(space + key + ": " + str(value))
    return config_str


def train_step_logger(path, epoch, train_loss, val_loss, train_metrics, val_metrics):
    with open(os.path.join(path, 'train_log.csv'), 'a') as file:
        line = str(epoch) + ',' + str(train_loss) + ',' + str(val_loss)
        for i in range(len(train_metrics)):
            line += ',' + str(train_metrics[i])
            line += ',' + str(val_metrics[i])
        file.write(line + '\n')
    file.close()


def test_logger(path, metrics, values):
    """
    creates a file 'test_log.txt' in the path containing for each line: metrics[i]: values[i]
    """
    with open(os.path.join(path, 'test_log.txt'), 'w') as f:
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(values[i]) + '\n')
