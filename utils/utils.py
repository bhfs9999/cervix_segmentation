import os
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(net, load_path, key_word=None, optimizer=None):
    # load_path = os.path.join(save_root, model_name)
    assert os.path.exists(load_path), 'Not exists requested checkpoint!'
    pre_model = torch.load(load_path)
    pre_dict = pre_model['state_dict']
    if key_word is not None:
        pre_dict = {k: v for k, v in pre_dict.items() if (key_word not in k)}
    else:
        pre_dict = {k: v for k, v in pre_dict.items()}
    state_dict = net.state_dict()
    # assert pre_dict.keys() == state_dict.keys()
    state_dict.update(pre_dict)
    net.load_state_dict(state_dict)
    print('Load Model From :{}'.format(load_path))
    return pre_model['iters'], pre_model['lr']


def save_model(net, optimizer, save_root, model_name, iters, lr, parallel=False):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    load_path = os.path.join(save_root, model_name)
    if parallel:
        save_dict = {
            'iters': iters,
            'state_dict': {k: v for k, v in net.module.state_dict().items() if ('criterion_loss' not in k)},
            'lr': lr,
            'optimizer_dict': optimizer.state_dict()
        }
    else:
        save_dict = {
            'iters': iters,
            'state_dict': {k: v for k, v in net.state_dict().items() if ('criterion_loss' not in k)},
            'lr': lr,
            'optimizer_dict': optimizer.state_dict()
        }
    torch.save(save_dict, load_path)