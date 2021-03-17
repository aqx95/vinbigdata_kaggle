import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn

from model import create_model
from dataset import prepare_testloader
from config import GlobalConfig


def inference(model, test_loader):
    model.eval()
    tbar = tqdm(enumerate(test_loader), total=len(test_loader))
    batch_pred = []
    for i, (images) in tbar:
        images = images.to(device)
        with torch.no_grad():
            logit_pred = model(images)
            pred = torch.sigmoid(logit_pred)
        batch_pred.append(pred.to('cpu'))
    batch_pred = np.concatenate(batch_pred, axis=0)
    return batch_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vinbigdata')
    parser.add_argument('--model', type=str, required=True, help='model architecture to be used')
    parser.add_argument('--model-name', type=str, required=True, help='pretrained model variant')
    args = parser.parse_args()

    config = GlobalConfig
    config.model = args.model
    config.model_name = args.model_name

    #read test csv
    test = pd.read_csv(os.path.join(config.CSV_PATH,'test.csv'))
    print("Shape: {}".format(test.shape))

    #prepare model and dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config).to(device)
    test_loader = prepare_testloader(test, config)

    #Inference
    print("Start inference...")
    model_ensemble = []
    for folds in range(config.num_folds):
        checkpoint_path = os.path.join(config.SAVE_PATH,f'{config.model_name}_fold{folds+1}.pt')
        print('Loading {} checkpoint'.format(checkpoint_path))
        states = torch.load(checkpoint_path)
        model.load_state_dict(states['model_state_dict'])
        model_ensemble += [inference(model, test_loader)]
    model_ensemble = np.mean(model_ensemble, axis=0)

    # submission
    test['label'] = model_ensemble
    test[['image_id', 'label']].to_csv('classification_sub.csv', index=False)
