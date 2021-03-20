import zipfile
import wget
import glob
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
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

    #overwrite
    config = GlobalConfig
    config.num_epochs = args.num_epochs
    config.image_size = args.image_size

    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.checkpoint_path):
      os.makedirs(config.checkpoint_path)

    logger = open(os.path.join(config.log_path, 'log.txt'), 'a')
    logger.write('Using GPU {} \n'.format(torch.cuda.get_device_name(0)))

    logger.write('Extracting train and test images...\n')
    #extract image data
    with zipfile.ZipFile(config.data_path, 'r') as zip_ref:
        zip_ref.extractall()
    logger.write('Extracting images DONE!\n')

    logger.write("Reading config from: {}\n".format(config.config_file))
    cfg = Config.fromfile(config.config_file)
    config.model_path = os.path.join(config.checkpoint_path, config.pretrained_model.split('/')[-1])
    logger.write("Downloading pretrained weights: {}\n".format(config.pretrained_model))
    wget.download(config.pretrained_model, config.model_path)

    ## Edit configuration settings
    cfg.classes = ("Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis")
    cfg.data.train.classes = cfg.classes
    cfg.data.val.classes = cfg.classes
    cfg.data.test.classes = cfg.classes

    cfg.data_root = 'train'
    cfg.data.train.img_prefix = cfg.data_root
    cfg.data.val.img_prefix = cfg.data_root
    cfg.data.test.img_prefix = cfg.data_root
    cfg.data.train.ann_file = '../../data/datacoco/annotation_1024_{}/instances_train2020.json'.format(args.fold_num)
    cfg.data.val.ann_file = '../../data/datacoco/annotation_1024_{}/instances_val2020.json'.format(args.fold_num)
    cfg.data.test.ann_file = '../../data/datacoco/annotation_1024_{}/instances_test2020.json'.format(args.fold_num)

    cfg.model.bbox_head.num_classes = 14

    cfg.data.samples_per_gpu = 4
    cfg.optimizer.lr = 0.0025
    cfg.evaluation.interval = 2
    cfg.checkpoint_config.interval = 2
    cfg.gpu_ids = range(1)
    cfg.seed = 0
    cfg.total_epochs = config.num_epochs
    cfg.runner.max_epochs = config.num_epochs

    cfg.load_from = config.model_path
    cfg.work_dir = config.output_path

    #Train
    logger.write('Begin training... \n')
    detector_train(cfg)

    ## Inference
    cfg.data.test.test_mode=True
    cfg.data.test.pipeline[0].type='LoadImageFromFile'
    cfg.model.test_cfg.score_thr = config.score_threshold
    config_file = cfg
    checkpoint_file = sorted(glob.glob(os.path.join(config.output_path,'epoch_*.pth')))[-1]
    logger.write('Read checkpoint at: {}\n'.format(checkpoint_file))

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model.CLASSES = cfg.classes

    logger.write('Creating submission...\n')
    submission_file = detector_test(model, config)
    submission_file.to_csv('submission.csv', index=False)
    logger.write('Finished!')

    logger.close()
