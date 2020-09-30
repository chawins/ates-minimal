'''Collection of utility and helper functions'''
import logging

import numpy as np
import torch


def get_logger(name, logger_name=None):
    # Get logger
    if logger_name is None:
        logger_name = name
    log_file = name + '.log'
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log


def classify(net, x, batch_size=200, num_classes=10):
    """Classify <x> with <net>."""
    with torch.no_grad():
        y_pred = torch.zeros((x.size(0), num_classes))
        for i in range(int(np.ceil(x.size(0) / batch_size))):
            begin = i * batch_size
            end = (i + 1) * batch_size
            y_pred[begin:end] = net(x[begin:end].to('cuda'))
    return y_pred


def get_acc(y_pred, y_test):
    """Compute accuracy based on network output (logits)."""
    return (y_pred.argmax(1) == y_test.to(y_pred.device)).float().mean().item()


def quantize(x, levels=16):
    """Quantization function from Qai et al. 2018 (CAT 2018)."""
    quant = torch.zeros_like(x)
    for i in range(1, levels):
        quant += (x >= i / levels).float()
    return quant / (levels - 1)
