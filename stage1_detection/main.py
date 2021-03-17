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


def detector_test(model, config):
    ## CSV reference
    test_meta = pd.read_csv(os.path.join(config.csv_path, 'test.csv'))
    sub = pd.read_csv(os.path.join(config.csv_path, 'sample_submission.csv'))
    # Loop imag for test
    for idx, row in tqdm(sub.iterrows(), total=len(sub), position=0, leave=True):
        img_id = row['image_id']
        meta = test_meta[test_meta['image_id']==img_id]
        orig_h, orig_w = meta['height'].item(), meta['width'].item()
        h_ratio, w_ratio = orig_h/1024, orig_w/1024

        img = mmcv.imread('test/'+img_id+'.png')
        result = inference_detector(model, img)
        string = ""
        for class_, class_array in enumerate(result):
            if class_array.shape[0]:
                class_array = class_array.tolist()
                for array in class_array:
                    array[0] = array[0] * w_ratio
                    array[1] = array[1] * h_ratio
                    array[2] = array[2] * w_ratio
                    array[3] = array[3] * h_ratio
                    string += '{} {:.2f} {} {} {} {} '.format(int(class_), array[4], int(array[0]), int(array[1]), int(array[2]), int(array[3]))
        if len(string) == 0:
            string += "14 1 0 0 1 1"

        sub.loc[idx, 'PredictionString'] = string

    return sub




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vinbigdata')
    parser.add_argument('--image-size', type=int, default=1024,
                        help='image size for training')
    parser.add_argument('--num-epochs', type=int, required=True,
                        help='number of training epoch')
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

    # logger.write('Extracting train and test images...\n')
    # #extract image data
    # with zipfile.ZipFile(config.data_path, 'r') as zip_ref:
    #     zip_ref.extractall()
    # logger.write('Extracting images DONE!\n')

    logger.write("Reading config from: {}\n".format(config.config_file))
    cfg = Config.fromfile(config.config_file)
    config.model_path = os.path.join(config.checkpoint_path, config.pretrained_model.split('/')[-1])
    logger.write("Downloading pretrained weights: {}\n".format(config.pretrained_model))
    wget.download(config.pretrained_model, config.model_path)

    ## Training
    cfg.total_epochs = config.num_epochs
    cfg.runner.max_epochs = config.num_epochs
    cfg.load_from = config.model_path
    cfg.work_dir = config.output_path
    cfg.data.train.ann_file = '../../data/datacoco/annotations_1024/instances_train2020.json'
    cfg.data.val.ann_file = '../../data/datacoco/annotations_1024/instances_val2020.json'
    cfg.data.test.ann_file = '../../data/datacoco/annotations_1024/instances_test2020.json'
    logger.write('Begin training... \n')
    #detector_train(cfg)

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
    submission.to_csv('submission.csv', index=False)
    logger.write('Finished!')

    logger.close()
