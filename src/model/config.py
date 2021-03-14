from pprint import pprint

__all__ = ['BaseConfig', 'TestConfig']

class BaseConfig:
    lesion_type = 'EX'

    train_img_path = '../../data/raw/1. Original Images/a. Training Set'
    train_mask_path = '../../data/raw/2. All Segmentation Groundtruths/a. Training Set/'
    augmentation = 'easy'

    finetune = True #Traning only decoder
    learning_rate = 1e-3
    num_epochs = 20
    batch_size = 8
    val_batch_size = 8
    is_fp16 = True
    weight_decay=1e-5

    model_name = "DeepLabV3Plus"
    
    model = {
    "encoder_name" : 'resnet18',
    "encoder_depth": 5,
    "encoder_weights" : 'imagenet',
    }

    metric = "iou"

    optimizer = "Adam"
    criterion = {"bce": 1.0, "log_dice": 1.0}
    scheduler = "reduce"
    mode = "max"

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
    test_img_paths = '../../data/raw/1. Original Images/b. Testing Set'
    test_mask_paths = '../../data/raw/2. All Segmentation Groundtruths/b. Testing Set'
    out_dir = '../../outputs'

if __name__ == '__main__':
    # pprint(TestConfig.get_all_attributes())
    d = dict(BaseConfig.__dict__).copy()
    d_1 = dict(BaseConfig.__base__.__dict__).copy()

    d.update(d_1)
    # dict(TestConfig.__dict__).update(dict(TestConfig.__base__.__dict__))
    pprint(d)
