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
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


def detector_test(model, config):
    ## CSV reference
    test_meta = pd.read_csv(os.path.join(config.csv_path, 'test.csv'))
    sub = pd.read_csv(os.path.join(config.csv_path, 'sample_submission.csv'))
    # Loop imag for test
    result_list = []
    for idx, row in tqdm(sub.iterrows(), total=len(sub), position=0, leave=True):
        img_id = row['image_id']
        meta = test_meta[test_meta['image_id']==img_id]
        orig_h, orig_w = meta['height'].item(), meta['width'].item()
        h_ratio, w_ratio = orig_h/config.image_size, orig_w/config.image_size

        img = mmcv.imread(os.path.join(config.test['test_root_path'], img_id+'.png'))
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

    #     row_result = []
    #     for class_, class_array in enumerate(result):
    #         if class_array.shape[0]:
    #             class_array = class_array.tolist()
    #             for array in class_array:
    #                 row_result = [img_id]
    #                 array[0] = array[0] * w_ratio
    #                 array[1] = array[1] * h_ratio
    #                 array[2] = array[2] * w_ratio
    #                 array[3] = array[3] * h_ratio
    #                 row_result.extend(int(class_), array[4], int(array[0]), int(array[1]), int(array[2]), int(array[3]))
    #                 result_list.append(row_result)
    #     if len(row_result) == 0:
    #         row_result.append([img_id, 14, 1, 0, 0, 1, 1])
    #
    # columns = ['image_id', 'class', 'score', 'x_min', 'y_min', 'x_max', 'y_max']
    # results_csv = pd.DataFrame(results_list, columns=columns)
    #
    # return results_csv



#Inference
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vinbigdata')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='checkpoint path for loading weights')
    parser.add_argument('--fold-num', type=int, default=0,
                        help='fold number for training')
    args = parser.parse_args()

    os.chdir('mmdetection')
    config = GlobalConfig
    config.fold_num = args.fold_num

    if not os.path.exists(config.log_path):
      os.makedirs(config.log_path)

    logger = Logger(config)
    logger.write("Reading config from: {}\n".format(config.config_file))
    base_cfg = Config.fromfile(config.config_file)

    ## Edit configuration settings
    model_config = overwrite_base(base_cfg, config, is_train=False)

    ## Inference
    config_file = model_config
    checkpoint_file = args.checkpoint_path
    logger.write('Read checkpoint at: {}\n'.format(checkpoint_file))

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model.CLASSES = config_file.classes

    logger.write('Creating submission...\n')
    submission_file = detector_test(model, config)
    submission_file.to_csv('../submission.csv', index=False)
    logger.write('Finished!')

    logger.close()
