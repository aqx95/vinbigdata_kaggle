'''
Convert CSV to COCO format
'''

import os
import json
import argparse
import pandas as pd
import numpy as np
import glob
import shutil


#mapper from class to id
class_to_id = {'Aortic enlargement':0, 'Atelectasis':1, 'Calcification':2, 'Cardiomegaly':3,
    		   	   'Consolidation':4, 'ILD':5, 'Infiltration':6, 'Lung Opacity':7, 'Nodule/Mass':8,
               	   'Other lesion':9, 'Pleural effusion':10, 'Pleural thickening':11, 'Pneumothorax':12,
               	   'Pulmonary fibrosis':13, 'No finding':14
               }


class Csv2Coco:
    def __init__(self, image_dir, total_annot, arg):
        self.images = []
        self.annotations = []
        self.categories = [] #dict of mapping {id:, name:}
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annot = total_annot
        self.arg = arg

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)

    # conversion to coco
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            shapes = self.total_annot[key]
            self.images.append(self._image(key, shapes[0]))
            for shape in shapes:
                bboxi = []
                for cor in shape[-7:-3]:
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
    def _image(self, path, shape):
        image = {}
        #print(path)
        #img = cv2.imread(self.image_dir + path + '.' + self.arg.file_type)
        image['height'] = shape[-2]#self.arg.image_size
        image['width'] = shape[-3]#self.arg.image_size
        image['id'] = path
        image['file_name'] = path + '.' + self.arg.file_type
        return image

    # 'annotations' field
    def _annotation(self, shape, label, path):
        points = shape
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = path
        annotation['category_id'] = label #int(class_to_id[str(label)])
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
    parser = argparse.ArgumentParser(description='VinBigData_trainval')
    parser.add_argument('--image-size', type=int, required=True, help='image size used for training')
    parser.add_argument('--fold-num', type=int, required=True, help='number of training folds')
    parser.add_argument('--file-type', type=str, required=True, help='image extension name')
    parser.add_argument('--save-path', type=str, default='datacoco', help='saved path')
    parser.add_argument('--csv-path', type=str, required=True, help='csv path for reading data')
    args = parser.parse_args()

    print(args)
    #read preprocessed train data
    csv_file = '../data/csv/' + args.csv_path
    image_dir = ''
    saved_coco_path = '../data/' + args.save_path

    annotations = pd.read_csv(csv_file)
    #rescale to training image size
    if args.csv_path != 'train_downsampled_fold.csv':
        annotations[['x_min','x_max']] = annotations[['x_min','x_max']].apply(lambda x:round(x*args.image_size,1))
        annotations[['y_min','y_max']] = annotations[['y_min','y_max']].apply(lambda x:round(x*args.image_size,1))

    for fold in range(args.fold_num):
        print('Fold {}...'.format(fold))
        total_train_annotations = {}
        total_val_annotations = {}
        train_annotation = annotations[annotations['fold'] != fold].values
        val_annotation = annotations[annotations['fold'] == fold].values

        for annotation in train_annotation:
            key = annotation[0].split(os.sep)[-1] #image_id
            value = np.array([annotation[1:]]) #remaining col
            if key in total_train_annotations.keys():
                total_train_annotations[key] = np.concatenate((total_train_annotations[key], value), axis=0)
            else:
                total_train_annotations[key] = value
        train_keys = list(total_train_annotations.keys())

        for annotation in val_annotation:
            key = annotation[0].split(os.sep)[-1] #image_id
            value = np.array([annotation[1:]]) #remaining col
            if key in total_val_annotations.keys():
                total_val_annotations[key] = np.concatenate((total_val_annotations[key], value), axis=0)
            else:
                total_val_annotations[key] = value
        val_keys = list(total_val_annotations.keys())

        print("train_n:", len(train_keys), 'val_n:', len(val_keys))

        #Create directory
        annot_path = os.path.join(saved_coco_path, 'annotation_{}_{}'.format(args.image_size, fold))
        if not os.path.exists(annot_path):
            os.makedirs(annot_path)
        # if not os.path.exists('%scoco/images/train2017/'%saved_coco_path):
        #     os.makedirs('%scoco/images/train2017/'%saved_coco_path)
        # if not os.path.exists('%scoco/images/val2017/'%saved_coco_path):
        #     os.makedirs('%scoco/images/val2017/'%saved_coco_path)

        #Convert CSV to Json
        print('Converting Trainset...')
        l2c_train = Csv2Coco(image_dir=image_dir, total_annot=total_train_annotations, arg=args)
        train_instance = l2c_train.to_coco(train_keys)
        l2c_train.save_coco_json(train_instance, os.path.join(annot_path,'instances_train2020.json'))
        # for file in train_keys:
        #     shutil.copy(image_dir+file+'.jpg',"%scoco/images/train2020/"%saved_coco_path)
        # for file in val_keys:
        #     shutil.copy(image_dir+file+'.jpg',"%scoco/images/val2020/"%saved_coco_path)
        print('Converting Valid set')
        l2c_val = Csv2Coco(image_dir=image_dir,total_annot=total_val_annotations, arg=args)
        val_instance = l2c_val.to_coco(val_keys)
        l2c_val.save_coco_json(val_instance, os.path.join(annot_path, 'instances_val2020.json'))
    print('COCO Conversion Done!')
