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
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


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


#Inference
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vinbigdata')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='checkpoint path for loading weights')
    parser.add_argument('--num-epochs', type=int, required=True,
                        help='number of training epoch')
    parser.add_argument('--fold-num', type=int, required=True,
                        help='fold number for training')
    args = parser.parse_args()

    os.chdir('mmdetection')

    #overwrite
    config = GlobalConfig
    config.num_epochs = args.num_epochs

    logger = open(os.path.join(config.log_path, 'infer_log.txt'), 'a')

    logger.write("Reading config from: {}\n".format(config.config_file))
    cfg = Config.fromfile(config.config_file)

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


    ## Inference
    cfg.data.test.test_mode=True
    cfg.data.test.pipeline[0].type='LoadImageFromFile'
    cfg.model.test_cfg.score_thr = config.score_threshold
    config_file = cfg
    checkpoint_file = args.checkpoint_path
    logger.write('Read checkpoint at: {}\n'.format(checkpoint_file))

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model.CLASSES = cfg.classes

    logger.write('Creating submission...\n')
    submission_file = detector_test(model, config)
    submission_file.to_csv('submission.csv', index=False)
    logger.write('Finished!')

    logger.close()
