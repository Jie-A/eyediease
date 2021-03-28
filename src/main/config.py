from pprint import pprint

__all__ = ['BaseConfig', 'TestConfig']


class BaseConfig:
    train_img_path = '../../data/raw/IDRiD/1. Original Images/a. Training Set'
    train_mask_path = '../../data/raw/IDRiD/2. All Segmentation Groundtruths/a. Training Set/'
    
    lesion_type = 'HE'
    dataset_name = 'IDRiD'
    data_mode = 'binary'
    augmentation = 'medium'
    scale_size = 1024
    input_format = 'all'  #2 type of input format : all image or patches

    #Final
    finetune = False  # Traning only decoder
    num_epochs = 50
    batch_size = 4
    val_batch_size = 4
    learning_rate = 1e-5
    learning_rate_decode = 1e-4
    weight_decay = 1e-5
    is_fp16 = True

    #first
    model_name = "UnetPlusPlus"
    model_params = {
        "encoder_name": 'efficientnet-b2',
        "encoder_depth": 5,
        "encoder_weights": "imagenet",
        "decoder_use_batchnorm": True,
        "decoder_attention_type": "scse",
        "classes": 1
    }

    #Choose at first and no need to change
    metric = "dice"
    mode = "max"

    #Second
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    # Should we use IOU loss instead of Dice loss in this case ?
    criterion = {"wbce": 1.0, "dice": 1.0}
    pos_weights = [200]
    optimizer = "diffgrad"
    scheduler = "cosr"

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
