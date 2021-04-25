from pprint import pprint
from pathlib import Path

__all__ = ['BaseConfig', 'TestConfig']

class BaseConfig:
    __basedir__ = 'data/raw/'
    dataset_name = 'IDRiD'
    train_img_path = Path(__basedir__) / dataset_name / '1. Original Images/a. Training Set'
    train_mask_path = Path(__basedir__) / dataset_name / '2. All Segmentation Groundtruths/a. Training Set'
    
    lesion_type = 'EX'
    data_mode = 'binary' 
    augmentation = 'advanced' #options: normal, easy, medium, advanced
    use_ben_transform = False #Good for vessel segmentation
    scale_size = 1024
    data_type = 'all'  #2 type of input format : whole image or tiles

    #Final
    finetune = False  # Traning only decoder
    num_epochs = 60
    batch_size = 2
    val_batch_size = 2
    learning_rate = 1e-5
    learning_rate_decode = 1e-3
    weight_decay = 2.5e-5
    is_fp16 = True

    #first
    model_name = "sa_unet"
    model_params= None
    #Choose at first and no need to change
    metric = "dice"
    mode = "max"

    #Second
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    # Should we use IOU loss instead of Dice loss in this case ?
    criterion = {"soft_bce": 0.8, 'symmetric_lovasz': 0.2}
    deep_supervision = False
    pos_weights = [200]
    optimizer = "madgrad"
    scheduler = "simple"

    resume_path = None #Resume training

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
    test_img_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / '1. Original Images/b. Testing Set'
    test_mask_path = Path(BaseConfig.__basedir__) / BaseConfig.dataset_name / '2. All Segmentation Groundtruths/b. Testing Set'
    out_dir = 'outputs'


if __name__ == '__main__':
    d = dict(BaseConfig.__dict__).copy()
    d_1 = dict(BaseConfig.__base__.__dict__).copy()
    pprint(d)