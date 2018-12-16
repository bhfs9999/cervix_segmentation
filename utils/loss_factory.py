from .loss import RobustFocalLoss2d, LovaszSigmoid, LovaszSoftmax, DiceLoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

__sets = {}

__sets['BCE'] = BCEWithLogitsLoss
__sets['cross_entropy'] = CrossEntropyLoss
__sets['Focal_loss'] = RobustFocalLoss2d
__sets['Lovasz_sigmoid'] = LovaszSigmoid
__sets['Lovasz_Softmax'] = LovaszSoftmax
__sets['dice_loss'] = DiceLoss


def get_loss(name):
    '''
    :param name:
    :return:
    '''
    if name not in __sets:
        raise KeyError('Unknown loss: {}'.format(name))
    else:
        return __sets[name]
