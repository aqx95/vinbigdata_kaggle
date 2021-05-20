import os

# Logger class
class Logger:
    def __init__(self, config, verbose=1):
        self.verbose = verbose
        self.logger = open(os.path.join(config.log_path, 'log.txt'), 'w')
    def write(self, msg):
        self.logger.write(msg)
        if self.verbose:
          print(msg)
    def close(self):
      self.logger.close()


# Overwrite base class with user-defined config
def overwrite_base(base, config, is_train=True):
    base.classes = config.classes
    base.data.train.classes = config.classes
    base.data.val.classes = config.classes
    base.data.test.classes = config.classes

    if is_train:
        base.data_root = config.train_root_path
    else:
        base.data_root = config.test['test_root_path']
    base.data.train.img_prefix = base.data_root
    base.data.val.img_prefix = base.data_root
    base.data.test.img_prefix = base.data_root
    base.data.train.ann_file = '../../data/datacoco/annotation_fold{}_1024/instances_train.json'.format(config.fold_num)
    base.data.val.ann_file = '../../data/datacoco/annotation_fold{}_1024/instances_val.json'.format(config.fold_num)
    base.data.test.ann_file = '../../data/datacoco/annotation_fold{}_1024/instances_test.json'.format(config.fold_num)

    base.model.bbox_head.num_classes = config.num_classes

    base.data.samples_per_gpu = config.samples_per_gpu
    base.optimizer.lr = config.lr
    base.evaluation.interval = config.eval_interval
    base.checkpoint_config.interval = config.checkpoint_interval
    base.gpu_ids = config.gpu
    base.seed = config.seed
    base.total_epochs = config.num_epochs
    base.runner.max_epochs = config.num_epochs

    base.load_from = config.model_path
    base.work_dir = config.output_path + '_' + str(config.fold_num)

    ### Test config
    if not is_train:
        base.data.test.test_mode = config.test['test_mode']
        base.data.test.pipeline[0].type = config.test['pipeline_type']
        base.model.test_cfg.score_thr = config.test['score_thr']

    return base
