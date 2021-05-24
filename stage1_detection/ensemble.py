import os
import glob
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from ensemble_boxes import *


def string_to_row(df, fold_num):
    csv = df.copy()
    data = {'image_id':[], 'model':None, 'class':[], 'confidence':[],
            'x_min':[], 'y_min':[], 'x_max':[], 'y_max': []}
    img_id = class_ = confidence = x_min = y_mix = x_max = y_max = []

    for idx, row in tqdm(csv.iterrows(), total=len(csv)):
        pred = row['PredictionString'].split(' ')
        for ptr in range(0, len(pred)-5, 6):
            data['image_id'].append(row['image_id'])
            data['class'].append(int(pred[ptr]))
            data['confidence'].append(float(pred[ptr+1]))
            data['x_min'].append(int(pred[ptr+2]))
            data['y_min'].append(int(pred[ptr+3]))
            data['x_max'].append(int(pred[ptr+4]))
            data['y_max'].append(int(pred[ptr+5]))

    data['model'] = fold_num
    new_df = pd.DataFrame(data, columns=[key for key in data.keys()])
    return new_df


def row_to_string(df):
    data = {'image_id':[], 'PredictionString':[]}
    current = df.iloc[0]['image_id']
    string = ""

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if row.image_id == current:
            string += f"{row['class']} {row.confidence} {row.x_min} {row.y_min} {row.x_max} {row.y_max} "
        else:
            data['image_id'].append(current)
            data['PredictionString'].append(string)
            #reset
            string = f"{row['class']} {row.confidence:.2f} {row.x_min} {row.y_min} {row.x_max} {row.y_max} "
            current = row.image_id
    #add last id
    data['image_id'].append(current)
    data['PredictionString'].append(string)

    new_df = pd.DataFrame(data, columns=[key for key in data.keys()])
    return new_df


# Weighted Box Fusion
def postprocess_fusion(df, fusion_type, iou_thr=0.5, sigma=0.1, skip_box_thr=0.0001):
    results = []
    image_ids = df["image_id"].unique()

    for image_id in tqdm(image_ids, total=len(image_ids), position=0, leave=True):
        # All annotations for the current image.
        data = df[df["image_id"] == image_id]
        data = data.reset_index(drop=True)

        annotations = {}
        weights = []

        # WBF expects the coordinates in 0-1 range.
        max_value = data.iloc[:, 4:].values.max()
        data.loc[:, ["x_min", "y_min", "x_max", "y_max"]] = data.iloc[:, 4:] / max_value #[4:] denotes x_min,y_min,x_max,y_max

        # Loop through all of the annotations for single image
        for idx, row in data.iterrows():
            model_id = row["model"]

            if model_id not in annotations:
                annotations[model_id] = {
                    "boxes_list": [],
                    "scores_list": [],
                    "labels_list": [],
                }

                # Assume equal weightage
                weights.append(1.0)

            annotations[model_id]["boxes_list"].append([row["x_min"], row["y_min"], row["x_max"], row["y_max"]])
            annotations[model_id]["scores_list"].append(row['confidence'])
            annotations[model_id]["labels_list"].append(row["class"])

        boxes_list = []
        scores_list = []
        labels_list = []
        #Combine all predcitions from all models for a image_id
        for annotator in annotations.keys():
            boxes_list.append(annotations[annotator]["boxes_list"])
            scores_list.append(annotations[annotator]["scores_list"])
            labels_list.append(annotations[annotator]["labels_list"])

        # Calculate Fusion
        if fusion_type == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr)

        if fusion_type == 'nms':
             boxes, scores, labels = nms(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_thr)

        if fusion_type == 'softnms':
            boxes, scores, labels = soft_nms(
                boxes_list,
                scores_list,
                labels_list,
                sigma=sigma,
                weights=weights,
                iou_thr=iou_thr)

        if fusion_type == 'nmw':
            boxes, scores, labels = non_maximum_weighted(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr)

        #Fused results for a single image
        for box, score, label in zip(boxes, scores, labels):
            results.append({
                "image_id": image_id,
                "class": int(label),
                "confidence": round(score, 2),
                "x_min": int(box[0] * max_value),
                "y_min": int(box[1] * max_value),
                "x_max": int(box[2] * max_value),
                "y_max": int(box[3] * max_value)
            })

    results = pd.DataFrame(results, columns=['image_id','class','confidence','x_min','y_min','x_max','y_max'])
    return results



#MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vinbigdata')
    parser.add_argument('--submission-path', type=str, required=True,
                        help='csv directory for single model detection')
    args = parser.parse_args()

    #Process each model submission
    ensemble_df = pd.DataFrame()
    for idx, subs in enumerate (glob.glob(args.submission_path + '/*.csv')):
        print('Read from {}'.format(subs.split('/')[-1]))
        model_df = pd.read_csv(subs)
        new_model_df = string_to_row(model_df, idx)
        ensemble_df = pd.concat([ensemble_df, new_model_df], axis=0)

    #Do WBF
    print("Shape of ensembled data before WBF: {}".format(ensemble_df.shape))
    wbf_ensemble = postprocess_fusion(ensemble_df, fusion_type='wbf')
    print("Shape of ensembled data after WBF: {}".format(wbf_ensemble.shape))

    #Convert back to submission format
    print("Converting to submission...")
    final_submission = row_to_string(wbf_ensemble)
    print("Shape of submission: {}".format(final_submission.shape))

    final_submission.to_csv(os.path.join(args.submission_path, 'ensemble_submission.csv'), index=False)
