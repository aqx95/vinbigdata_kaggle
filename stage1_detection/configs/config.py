class GlobalConfig:

    image_size = 1024

    #model setting
    config_file = 'configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
    pretrain_url = 'https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth'

    #directory
    csv_path = '../../data/csv'
    data_path = '/content/drive/Shareddrives/Deep Learning/vinbigdata_1024x1024_png.zip'
    output_path = '../vinbig_output'
    log_path = '../logs'
    pretrain_store_path = '../checkpoint'


    ### Overwrite model configuration
    # Data
    num_classes  = 14
    classes = ("Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
                "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
                "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
                "Pulmonary fibrosis")

    # Path
    train_root_path = '../../../train'
    model_path = None

    # Training
    samples_per_gpu = 4
    lr = 0.0025
    seed = 0
    num_epochs = 20
    gpu = [0]

    # Logs
    checkpoint_interval = 2
    eval_interval = 2

    test = {
        'test_mode': True,
        'pipeline_type': 'LoadImageFromFile',
        'score_thr': 0.15
    }
