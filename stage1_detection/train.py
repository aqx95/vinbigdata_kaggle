import zipfile
import wget
import glob
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm

from utils import overwrite_base, Logger
from config import GlobalConfig

import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


def detector_train(cfg):
    model = build_detector(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vinbigdata')
    parser.add_argument('--image-size', type=int, default=1024,
                        help='image size for training')
    parser.add_argument('--num-epochs', type=int, required=True,
                        help='number of training epoch')
    parser.add_argument('--fold-num', type=int, required=True,
                        help='fold number for training')
    args = parser.parse_args()

    os.chdir('mmdetection')

    #Overwrite
    config = GlobalConfig
    config.num_epochs = args.num_epochs
    config.image_size = args.image_size
    config.fold_num = args.fold_num

    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    #Init logger
    logger = Logger(config)
    logger.write('Using GPU {} \n'.format(torch.cuda.get_device_name(0)))

    #Read base config file
    logger.write("Reading config from: {}".format(config.config_file))
    base_cfg = Config.fromfile(config.config_file)
    #Download pretrained model
    config.model_path = os.path.join(config.pretrain_store_path, config.pretrain_url.split('/')[-1])
    if not os.path.exists(config.pretrain_store_path):
      os.makedirs(config.pretrain_store_path)
      logger.write("Downloading pretrained weights: {}\n".format(config.pretrain_url))
      wget.download(config.pretrain_url, config.model_path)
    else:
      logger.write("Pretrained model already in cache \n")

    # Edit configuration settings
    final_config = overwrite_base(base_cfg, config, is_train=True)
    with open(os.path.join(config.output_path, config.model_name+'.py'), 'w') as f:
        f.write(final_config.pretty_text)

    #Train
    logger.write(f'Begin training Fold {config.fold_num}... \n')
    detector_train(final_config)
    logger.write(f'Finished training Fold {config.fold_num}! \n')
    logger.close()
