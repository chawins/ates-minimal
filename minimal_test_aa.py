'''Test CIFAR-10 model'''
from __future__ import print_function

import datetime
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from autoattack import AutoAttack

from lib.dataset_utils import (load_cifar10_all, load_cifar100_all,
                               load_mnist_all)
from lib.mnist_model import BasicModel
from lib.utils import classify, get_acc, get_logger
from lib.wideresnet import WideResNet


def main(config):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']
    model_name = config['meta']['model_name']

    # Set all random seeds
    seed = config['meta']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    # Set up model directory
    save_dir = os.path.join(config['meta']['save_path'], './')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    timenow = datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    logfile = 'test_' + model_name + '_' + timenow
    log = get_logger(logfile, '')
    log.info('\n%s', yaml.dump(config))
    log.info('Preparing data...')
    if config['test']['dataset'] == 'cifar10':
        _, _, (x_test, y_test) = load_cifar10_all(
            data_dir=config['meta']['data_path'], val_size=0.1, shuffle=False,
            seed=seed)
        num_classes = 10
    elif config['test']['dataset'] == 'cifar100':
        _, _, (x_test, y_test) = load_cifar100_all(
            data_dir=config['meta']['data_path'], val_size=0.1, shuffle=False,
            seed=seed)
        num_classes = 100
    elif config['test']['dataset'] == 'mnist':
        _, _, (x_test, y_test) = load_mnist_all(
            data_dir=config['meta']['data_path'], val_size=0.1, shuffle=True,
            seed=seed)
        num_classes = 10
    else:
        raise NotImplementedError('invalid dataset.')

    log.info('Building model...')
    if config['test']['network'] == 'resnet':
        # use ResNetV2-20
        net = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif config['test']['network'] == 'wideresnet':
        # use WideResNet-34-10
        net = WideResNet(num_classes=num_classes)
    elif config['test']['network'] == 'basic':
        net = BasicModel(num_classes=num_classes)
    else:
        raise NotImplementedError('Specified network not implemented.')
    net.load_state_dict(torch.load(model_path + '.pt'))
    net = net.eval().to(device)

    num_test_samples = config['test']['num_test_samples']
    log.info('Starting attack...')
    adversary = AutoAttack(
        net, norm='Linf', eps=config['test']['epsilon'], version='standard',
        log_path=logfile + '.log')
    x_adv = adversary.run_standard_evaluation(x_test[:num_test_samples],
                                              y_test[:num_test_samples],
                                              bs=config['test']['batch_size'])
    y_pred = classify(net, x_adv)
    adv_acc = get_acc(y_pred, y_test[:num_test_samples])
    log.info('AA acc: %.4f.', adv_acc)


if __name__ == '__main__':
    # Uncomment for MNIST test
    # if not os.path.exists('./mnist-03.pt'):
    #     os.popen(
    #         'curl -L https://berkeley.box.com/shared/static/gx9il4gakesod7p7lclxvm01ynadcxh9.pt --output mnist-03.pt')
    # config = {
    #     'meta': {
    #         'model_name': 'mnist-03',
    #         'save_path': './',
    #         'data_path': '~/mount/',
    #         'seed': 2019,
    #         'gpu_id': '7',
    #     },
    #     'test': {
    #         'dataset': 'mnist',
    #         'network': 'basic',
    #         'batch_size': 500,
    #         'num_test_samples': 10000,
    #         'epsilon': 0.3,
    #     }
    # }

    # Uncomment for CIFAR-10 test
    if not os.path.exists('./cifar10-wrn.pt'):
        os.popen(
            'curl -L https://berkeley.box.com/shared/static/hgehswlmggrs5si9ziia2uuquxtngvd2.pt --output cifar10-wrn.pt')
    config = {
        'meta': {
            'model_name': 'cifar10-wrn',
            'save_path': './',
            'data_path': '~/mount/',
            'seed': 2019,
            'gpu_id': '5',
        },
        'test': {
            'dataset': 'cifar10',
            'network': 'wideresnet',
            'batch_size': 200,
            'num_test_samples': 10000,
            'epsilon': 8 / 255,
        }
    }

    # Uncomment for CIFAR-100 test
    # if not os.path.exists('./cifar100-wrn.pt'):
    #     os.popen('curl -L https://berkeley.box.com/shared/static/b1t0613tw7bhkvojkihijqghebxsqcpf.pt --output cifar100-wrn.pt')
    # config = {
    #     'meta': {
    #         'model_name': 'cifar100-wrn',
    #         'save_path': './',
    #         'data_path': '~/mount/',
    #         'seed': 2019,
    #         'gpu_id': '7',
    #     },
    #     'test': {
    #         'dataset': 'cifar100',
    #         'network': 'wideresnet',
    #         'batch_size': 200,
    #         'num_test_samples': 10000,
    #         'epsilon': 8 / 255,
    #     }
    # }

    main(config)
