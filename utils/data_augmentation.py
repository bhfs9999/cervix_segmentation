from .transforms import *

def get_train_augmentation(mean, std, up_rate=2, target_size=256):
    return  Compose(
        [
            HorizontalFlip(),
            OneOf([
                RandomShiftCrop(p=0.3),
                # ElasticTransform(alpha_affine=10, p=0.4, interpolation=cv2.INTER_NEAREST),
                ShiftScaleRotate(p=0.4),
            ], p=0.5),
            OneOf([
                RandomBrightnessShift(p=0.3),
                RandomBrightnessMultiply(p=0.3),
                RandomBrightnessGamma(p=0.3),
            ], p=0.5),
            # ReflectiveCenterPad(up_rate=up_rate, target_size=target_size, p=1),
            Normalize(mean=mean, std=std),
            Resize(size=target_size),
            ToTensor()
        ])

def get_valid_normalization(mean, std, up_rate=2, target_size=256):
    return Compose(
        [
            # ReflectiveCenterPad(up_rate=up_rate, target_size=target_size, p=1),
            Normalize(mean=mean, std=std),
            Resize(size=target_size),
            ToTensor()
        ])


class TrainAugmentation(object):
    def __init__(self, mean, std, up_rate=2, target_size=256):
        super(TrainAugmentation, self).__init__()
        self.transform = Compose([
            HorizontalFlip(),
            OneOf([
                RandomShiftCrop(p=0.3),
                ElasticTransform(alpha_affine=10, p=0.4),
                ShiftScaleRotate(p=0.4),
            ], p=0.5),
            OneOf([
                RandomBrightnessShift(p=0.3),
                RandomBrightnessMultiply(p=0.3),
                RandomBrightnessGamma(p=0.3),
            ], p=0.5),
            ReflectiveCenterPad(up_rate=up_rate, target_size=target_size, p=1),
            Normalize(mean=mean, std=std),
            Resize(size=target_size),
            ToTensor()
        ])

    def __call__(self, input_data):
        return self.transform(**input_data)

class TestNormalization(object):
    def __init__(self, mean, std, up_rate=2, target_size=256):
        super(TestNormalization, self).__init__()
        self.transform = Compose([
            ReflectiveCenterPad(up_rate=up_rate, target_size=target_size, p=1),
            Normalize(mean=mean, std=std),
            Resize(size=target_size),
            ToTensor()
        ])

    def __call__(self, input_data):
        return self.transform(**input_data)