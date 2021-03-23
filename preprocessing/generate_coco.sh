#!/bin/sh

#Train
python csv2coco_train_val.py --image-size 2000 --fold-num 5 --file-type jpg \
                             --csv-path train_downsampled_fold.csv --image-dir ../../train/
