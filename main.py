import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.train import train
from src.test import evaluate


def load_config(path='configs/config.yaml'):
    stream = open(path, 'r')
    return edict(yaml.safe_load(stream))


def __train(config_path):
    config = load_config(path=config_path)
    train(config)


def __test(path):
    config = load_config(path=os.path.join(path, 'config.yaml'))
    evaluate(path, config)


def main(options):
    """
    launches a training or a test or a prediction depending on the chosen mode.
    To run a training, set --mode train and --config <path_to_your_config>.
        By default, the path_to_your_config = 'configs/configs.yaml'.
        The training will run, and you can find the results in the log folder.
    To perform a test or a prediction, put --mode 'test' or 'predict' and -- path <your_path>
        which is a folder containing: 'config.yaml' and the model weights (.h5 file)
        this is done in such a way that if you train a model with a certain config that is going to be stored in
        'logs/experiment_1', to test or predict it, you just have to put --path logs/experiment_1
    """
    if options['mode'] == 'train':
        __train(options['config_path'])

    elif options['mode'] == 'test':
        __test(options['path'])

    # elif options['mode'] == 'predict':
    #     __prediction(options['path'])

    else:
        raise "choose a mode between 'train', 'test' and 'predict'"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'test' and 'predict'")
    parser.add_argument('--config_path', default='configs\\config.yaml', type=str, help="path to config (just for training)")
    parser.add_argument('--path', type=str, help="experiment path")

    args = parser.parse_args()
    options = vars(args)

    main(options)