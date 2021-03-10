'''
Convert CSV to COCO format
'''

import os
import json
import pandas as pd
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
np.random.seed(2020)

#mapper from class to id
class_to_id = {'Aortic enlargement':1, 'Atelectasis':2, 'Calcification':3, 'Cardiomegaly':4,
    		   	   'Consolidation':5, 'ILD':6, 'Infiltration':7, 'Lung Opacity':8, 'Nodule/Mass':9,
               	   'Other lesion':10, 'Pleural effusion':11, 'Pleural thickening':12, 'Pneumothorax':13,
               	   'Pulmonary fibrosis':14
               }

class Csv2Coco:
    def __init__(self, image_dir, total_annot):
        self.images = []
        self.annotations = []
        self.categories = [] #dict of mapping {id:, name:}
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annot = total_annot

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)

    # conversion to coco
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annot[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[-4:]:
                    bboxi.append(int(float(cor)))
                label = shape[0]
                annotation = self._annotation(bboxi,label,key)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'https://github.com/Klawens/dataset_prepare.git'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 'categories' field
    def _init_categories(self):
        for k, v in class_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 'images' field
    def _image(self, path):
        image = {}
        #print(path)
        #img = cv2.imread(self.image_dir + path + '.jpg')
        image['height'] = 512#img.shape[0]
        image['width'] = 512#img.shape[1]
        image['id'] = path
        image['file_name'] = path + '.png'
        return image

    # 'annotations' field
    def _annotation(self, shape, label, path):
        points = shape
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = path
        annotation['category_id'] = int(class_to_id[str(label)])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # CoCo format [x1,y1,h,w]
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)

    # 'annotations: {segmentation}' field
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a


if __name__ == '__main__':
    csv_file = 'data/final_train.csv'
    image_dir = ''
    saved_coco_path = 'data'

    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file, header=None, skiprows=1).values
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1] #image_id
        value = np.array([annotation[1:]]) #remaining col
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value

    total_keys = list(total_csv_annotations.keys())
    train_keys, val_keys = train_test_split(total_keys, test_size=0.2)
    print("train_n:", len(train_keys), 'val_n:', len(val_keys))


    #Create directory
    if not os.path.exists('%scoco/annotations/'%saved_coco_path):
        os.makedirs('%scoco/annotations/'%saved_coco_path)
    # if not os.path.exists('%scoco/images/train2017/'%saved_coco_path):
    #     os.makedirs('%scoco/images/train2017/'%saved_coco_path)
    # if not os.path.exists('%scoco/images/val2017/'%saved_coco_path):
    #     os.makedirs('%scoco/images/val2017/'%saved_coco_path)

    #Convert CSV to Json
    print('Converting Trainset...')
    l2c_train = Csv2Coco(image_dir=image_dir, total_annot=total_csv_annotations)
    train_instance = l2c_train.to_coco(total_keys)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2020.json'%saved_coco_path)
    # for file in train_keys:
    #     shutil.copy(image_dir+file+'.jpg',"%scoco/images/train2020/"%saved_coco_path)
    # for file in val_keys:
    #     shutil.copy(image_dir+file+'.jpg',"%scoco/images/val2020/"%saved_coco_path)
    print('Converting Valid set')
    l2c_val = Csv2Coco(image_dir=image_dir,total_annot=total_csv_annotations)
    val_instance = l2c_val.to_coco(val_keys)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2020.json'%saved_coco_path)
    print('COCO Conversion Done!')
