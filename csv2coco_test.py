'''
Convert CSV to COCO (test)
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import os
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split
np.random.seed(2020)

classname_to_id = {'Aortic enlargement':1, 'Atelectasis':2, 'Calcification':3, 'Cardiomegaly':4,
    		   	   'Consolidation':5, 'ILD':6, 'Infiltration':7, 'Lung Opacity':8, 'Nodule/Mass':9,
               	   'Other lesion':10, 'Pleural effusion':11, 'Pleural thickening':12, 'Pneumothorax':13,
               	   'Pulmonary fibrosis':14
                   }

class Csv2Coco:
    def __init__(self, img_dir, total_img):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.img_dir = img_dir
        self.total_img = total_img

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path,'w'), ensure_ascii=False, indent=2)

    def to_coco(self):
        self._init_categories()
        for key in self.total_img:
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
        image['height'] = 512
        image['width'] = 512
        image['id'] = path
        image['file_name'] = path + '.png'
        return image


if __name__ == '__main__':
    csv_file = 'data/sample_submission.csv'
    image_dir = ''
    saved_coco_path = 'data'

    total_img = []
    test_rows = pd.read_csv(csv_file, header=None, skiprows=1).values
    for row in test_rows:
        test_img = row[0].split(os.sep)[-1] #image_id
        total_img.append(test_img)

    #Convert test csv to json
    print('Converting Testset...')
    l2c_test = Csv2Coco(img_dir=image_dir, total_img=total_img)
    test_instance = l2c_test.to_coco()
    l2c_test.save_coco_json(test_instance, '%scoco/annotations/instances_test2020.json'%saved_coco_path)
