'''
Convert CSV to COCO (test)
'''

import os
import json
import argparse
import numpy as np
import pandas as pd
import glob
import os
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split

classname_to_id = {'Aortic enlargement':0, 'Atelectasis':1, 'Calcification':2, 'Cardiomegaly':3,
    		   	   'Consolidation':4, 'ILD':5, 'Infiltration':6, 'Lung Opacity':7, 'Nodule/Mass':8,
               	   'Other lesion':9, 'Pleural effusion':10, 'Pleural thickening':11, 'Pneumothorax':12,
               	   'Pulmonary fibrosis':13
               }

class Csv2Coco:
    def __init__(self, img_dir, total_annot, arg):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.img_dir = img_dir
        self.total_annot = total_annot
        self.arg = arg

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path,'w'), ensure_ascii=False, indent=2)

    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            shape = self.total_annot[key]
            self.images.append(self._image(key))
            self.img_id += 1
        instance = {}
        instance['info'] = 'AQX'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        for k, v in classname_to_id.items():
            categories = {}
            categories['id'] = v
            categories['name'] = k
            self.categories.append(categories)

    def _image(self, path):
        image = {}
        image['height'] = self.arg.image_size
        image['width'] = self.arg.image_size
        image['id'] = path
        image['file_name'] = path + '.' + self.arg.file_type
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VinBigData_test')
    parser.add_argument('--image-size', type=int, default=1024, help='image size used for training')
    parser.add_argument('--file-type', type=str, default='png', help='image extension name')
    parser.add_argument('--fold-num', type=int, default=5, help='number of training folds')
    parser.add_argument('--save-path', type=str, default='datacoco', help='saved path')
    args = parser.parse_args()

    print(args)
    #read test data
    csv_file = '../data/csv/test.csv'
    image_dir = ''
    saved_coco_path = '../data/' + args.save_path

    total_annotation = {}
    test_rows = pd.read_csv(csv_file, header=None, skiprows=1).values
    for row in test_rows:
        test_key = row[0].split(os.sep)[-1] #image_id
        value = np.array(row[1:])
        total_annotation[test_key] = value
    test_keys = list(total_annotation.keys())
    print('Number of test: {}'.format(len(test_keys)))

    for fold in range(args.fold_num):
        print('Fold {}...'.format(fold))
        annot_path = os.path.join(saved_coco_path, 'annotation_fold{}_{}'.format(fold, args.image_size))
        #Convert test csv to json
        print('Converting Testset...')
        l2c_test = Csv2Coco(img_dir=image_dir, total_annot=total_annotation, arg=args)
        test_instance = l2c_test.to_coco(test_keys)
        l2c_test.save_coco_json(test_instance,  os.path.join(annot_path, 'instances_test.json'))
