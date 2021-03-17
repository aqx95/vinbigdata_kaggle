class GlobalConfig:
    seed = 2020
    num_classes = 1
    batch_size = 16
    num_epochs = 20
    image_size = 300
    verbose = 1
    verbose_step = 1
    num_folds = 5
    train_one_fold = False

    # unpack the key dict
    scheduler = 'CosineAnnealingWarmRestarts'
    scheduler_params = {'StepLR': {'step_size':2, 'gamma':0.3, 'last_epoch':-1, 'verbose':True},

                'ReduceLROnPlateau': {'mode':'max', 'factor':0.5, 'patience':0, 'threshold':0.0001,
                                      'threshold_mode':'rel', 'cooldown':0, 'min_lr':0,
                                      'eps':1e-08, 'verbose':True},

                'CosineAnnealingWarmRestarts': {'T_0':20, 'T_mult':1, 'eta_min':1e-6, 'last_epoch':-1,
                                                'verbose':True}, #train step

                'CosineAnnealingLR':{'T_max':20, 'last_epoch':-1} #validation step
                }

    # do scheduler.step after optimizer.step
    train_step_scheduler = True
    val_step_scheduler = False

    # optimizer
    optimizer = 'Adam'
    optimizer_params = {'AdamW':{'lr':1e-4, 'betas':(0.9,0.999), 'eps':1e-08,
                                 'weight_decay':1e-6,'amsgrad':False},
                        'SGD':{'lr':0.001, 'momentum':0., 'weight_decay':0.01},
                        'Adam':{'lr':1e-4, 'weight_decay':1e-5}
                        }

    # criterion
    criterion = "bce"
    criterion_params = {'crossentropy': {'weight':None,'size_average':None,
                                             'ignore_index':-100,'reduce':None,
                                             'reduction':'mean'},
                        'labelsmoothloss': {'num_class':2, 'smoothing':0.3, 'dim':-1},}

    IMAGE_COL = 'image_id'
    TARGET_COL = 'label'
    SAVE_PATH = 'save'
    LOG_PATH = 'log'
    CSV_PATH = '../data/csv'
    TRAIN_PATH = '../../train'
    TEST_PATH = '../../test'

    model = 'effnet'
    model_name = 'tf_efficientnet_b3'
    drop_rate = 0.0
    pretrained = True
