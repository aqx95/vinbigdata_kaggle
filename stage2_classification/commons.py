import os
import logging
import torch
import sklearn
import numpy as np
from sklearn.metrics import roc_auc_score


def log(config, name):
    log_file = os.path.join(config.LOG_PATH, 'log.txt')
    if not os.path.isfile(log_file):
        os.makedirs(config.LOG_PATH)
        open(log_file, "w+").close()

    console_log_format = "%(levelname)s %(message)s"
    file_log_format = "%(levelname)s: %(asctime)s: %(message)s"

    #Configure logger
    logging.basicConfig(level=logging.INFO, format=console_log_format)
    logger = logging.getLogger(name)

    #File handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(file_log_format)
    handler.setFormatter(formatter)

    #Stream handler
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(file_log_format)
    s_handler.setFormatter(formatter)

    #Add handler to logger
    logger.addHandler(handler)
    logger.addHandler(s_handler)

    return logger


class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.count = 0

    def update(self, batch_loss, batch_size):
        self.loss += batch_loss*batch_size
        self.count += batch_size

    @property
    def avg(self):
        return self.loss/self.count


class AucMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pred = []
        self.target = []

    def update(self, pred, target):
        self.pred += [pred.sigmoid().cpu()]
        self.target += [target.detach().cpu()]

    def macro_auc(self, pred, label):
        aucs = []
        for i in range(label.shape[1]):
            aucs.append(roc_auc_score(label[:, i], pred[:, i]))
        return np.mean(aucs)

    @property
    def get_auc(self):
        self.pred = torch.cat(self.pred).numpy()
        self.target = torch.cat(self.target).numpy()
        return self.macro_auc(self.pred, self.target)
