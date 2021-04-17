from pprint import pprint

__all__ = ['BaseConfig', 'TestConfig']


class BaseConfig:
    train_img_path = '../../data/raw/IDRiD/1. Original Images/a. Training Set'
    train_mask_path = '../../data/raw/IDRiD/2. All Segmentation Groundtruths/a. Training Set/'
    
    lesion_type = 'MA'
    dataset_name = 'IDRiD'
    data_mode = 'binary'
    augmentation = 'medium'
    use_ben_transform = False
    scale_size = 256
    data_type = 'tile'  #2 type of input format : whole image or patches

    #Final
    finetune = False  # Traning only decoder
    num_epochs = 60
    batch_size = 8
    val_batch_size = 8
    learning_rate = 1e-3
    learning_rate_decode = 1e-3
    weight_decay = 2.5e-5
    is_fp16 = False

    #first
    model_name = "efficientnetb2_attunet"
    model_params =  {"num_classes": 1, 
    "drop_rate": 0.2, 
    "pretrained": True, 
    "freeze_bn": False, 
    "freeze_backbone": False}

    #Choose at first and no need to change
    metric = "auc_pr"
    mode = "max"

    #Second
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    # Should we use IOU loss instead of Dice loss in this case ?
    criterion = {"bce": 0.8, "log_dice": 0.2}
    deep_supervision = False
    pos_weights = [200]
    optimizer = "madgrad"
    scheduler = "simple"

    resume_path = "../../models/MA/Apr17_11_42/checkpoints/best_full.pth"

    @classmethod
    def get_all_attributes(cls):
        d = {}
        son = dict(cls.__dict__)
        dad = dict(cls.__base__.__dict__)

        son.update(dad)
        for k, v in son.items():
            if not k.startswith('__') and k != 'get_all_attributes':
                d[k] = v

        return d


class TestConfig(BaseConfig):
    test_img_paths = '../../data/raw/IDRiD/1. Original Images/b. Testing Set'
    test_mask_paths = '../../data/raw/IDRiD/2. All Segmentation Groundtruths/b. Testing Set'
    out_dir = '../../outputs'
    out_figures = out_dir + '/figures'


if __name__ == '__main__':
    d = dict(BaseConfig.__dict__).copy()
    d_1 = dict(BaseConfig.__base__.__dict__).copy()
    pprint(d)
