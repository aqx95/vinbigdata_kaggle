class GlobalConfig:
    image_size = 1024
    num_epochs = 20
    score_threshold = 0.15

    #model setting
    config_file = 'configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
    pretrained_model = 'https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth'

    #paths
    csv_path = '../../data/csv'
    data_path = '/content/drive/Shareddrives/Deep Learning/vinbigdata_1024x1024_png.zip'
    checkpoint_path = 'checkpoint'
    model_path = None
    output_path = 'vinbig_output'
    log_path = 'logs'
    train_root_path = '../../../train'

    #overwrite
    classes = ("Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
                "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
                "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
                "Pulmonary fibrosis")
