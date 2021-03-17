import zipfile
import wget
import glob
import os
import argparse
import pandas as pd
from tqdm import tqdm

from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


def train_detector(cfg):
    model = build_detector(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    train_detector(model, datasets, cfg, distributed=False, validate=True)


def test_detector(model, config):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=1024, help='image size for training')
    parser.add_argument('--num-epochs', type=int, required=True, help='number of training epoch')

    os.chdir('mmdetection')

    with zipfile.ZipFile(config.data_path, 'r') as zip_ref:
        zip_ref.extractall()

    cfg = Config.fromfile(config.config_file)
    config.model_path = os.path.join(config.checkpoint_path, config.pretrained_model.split('/')[-1])

    if not os.path.exists(config.checkpoint_path):
        wget.download(config.pretrained_model, config.model_path)

    ## Training
    cfg.load_from = config.checkpoint_path
    cfg.work_dir = config.output_path
    train_detector(cfg)

    ## Inference
    cfg.data.test.test_mode=True
    cfg.data.test.pipeline[0].type='LoadImageFromFile'
    cfg.model.test_cfg.score_thr = config.score_threshold
    config_file = cfg
    checkpoint_file = glob.glob(os.path.join(config.output_path,'epoch_*.pth'))[-1]

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model.CLASSES = cfg.classes

    submission_file = test_detector(model, config)
    submission.to_csv('submission.csv', index=False)
