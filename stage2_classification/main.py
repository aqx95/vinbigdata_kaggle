import torch
import os
import pandas as pd
import numpy as np
import yaml
import seaborn as sns
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import sklearn
import torch.nn.functional as F
from glob import glob
from skimage import io
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import GlobalConfig
from model import create_model
from commons import log
from engine import Fitter
from dataset import prepare_loader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_single_fold(df_folds, config, device, fold):
    model = create_model(config).to(device)
    train_df = df_folds[df_folds['fold'] != fold].reset_index(drop=True)
    valid_df = df_folds[df_folds['fold'] == fold].reset_index(drop=True)
    fold_target = np.asarray(valid_df[config.TARGET_COL].values)

    train_loader, valid_loader = prepare_loader(train_df, valid_df, config)
    fitter = Fitter(model, device, config)
    logger.info("Fold {} data preparation DONE...".format(fold))
    best_checkpoint = fitter.fit(train_loader, valid_loader, fold)
    valid_df['oof_pred'] = best_checkpoint['oof_pred']
    logger.info("Finish Training Fold {}".format(fold))

    return valid_df


def train_loop(df_folds: pd.DataFrame, config, device, fold_num:int=None, train_one_fold=False):
    target = []
    oof_pred = []
    oof_df = pd.DataFrame()

    if train_one_fold:
        _oof_df = train_single_fold(df_folds=df_folds, config=config, device=device, fold=fold_num)
        oof_df = pd.concat([oof_df, _oof_df])
        curr_fold_auc = sklearn.metrics.roc_auc_score(_oof_df['label'], _oof_df['oof_pred'])
        logger.info("Fold {} AUC Score: {}".format(fold_num, curr_fold_auc))

    else:
        for fold in (number+1 for number in range(config.num_folds)):
            _oof_df = train_single_fold(df_folds=df_folds, config=config, device=device, fold=fold)
            oof_df = pd.concat([oof_df, _oof_df])
            curr_fold_auc = sklearn.metrics.roc_auc_score(_oof_df['label'], _oof_df['oof_pred'])
            logger.info("Fold {} AUC Score: {}".format(fold, curr_fold_auc))
            logger.info("-------------------------------------------------------------------")

        oof_auc = sklearn.metrics.roc_auc_score(oof_df['label'], oof_df['oof_pred'])
        logger.info("5 Folds OOF AUC Score: {}".format(oof_auc))
        oof_df.to_csv(f"oof_{config.model_name}.csv")


## Main run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vinbigdata')
    parser.add_argument('--num-epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--image-size', type=int, default=512, help='image size for training')
    parser.add_argument('--train-one-fold', type=bool, default=False, help='train one/all folds')
    parser.add_argument('--model', type=str, required=True, help='model architecture to be used')
    parser.add_argument('--model-name', type=str, required=True, help='pretrained model variant')
    args = parser.parse_args()

    #overwrite settings
    config = GlobalConfig
    config.num_epochs = args.num_epochs
    config.image_size = args.image_size
    config.train_one_fold = args.train_one_fold
    config.model = args.model
    config.model_name = args.model_name

    seed_everything(config.seed)

    #initialise logger
    logger = log(config, 'training')
    logger.info(config.__dict__)
    logger.info("-------------------------------------------------------------------")

    train = pd.read_csv(os.path.join(config.CSV_PATH,'train.csv'))
    logger.info(f'Shape: {train.shape} | Number of unique: {train.image_id.nunique()}')

    # convert to labels 0 (normal) and 1 (abnormal)
    train['label'] = -1
    train.loc[train.class_id==14, 'label'] = 0
    train.loc[train.class_id!=14, 'label'] = 1

    train.drop_duplicates(subset=['image_id'], inplace=True)
    train = train[['image_id','label']].reset_index(drop=True)
    logger.info(f'Shape: {train.shape} | Number of unique: {train.image_id.nunique()}')
    logger.info("-------------------------------------------------------------------")

    #split fold
    train['fold'] = -1
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y=train[config.TARGET_COL])):
        train.loc[val_idx, 'fold'] = fold+1

    #training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loop(df_folds=train, config=config, device=device, fold_num=1,
               train_one_fold= config.train_one_fold)
