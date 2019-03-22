import numpy as np
import torch
import torch.nn as nn
import pdb

from sklearn import metrics
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = 2 * self.inter.float() / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    # print(input.size(), target.size())
    valid_num = 0
    assert input.size() == target.size()
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        if c[1].max().data < 1.:
            continue
        s = s + DiceCoeff().forward(c[0], c[1])
        valid_num += 1
    if valid_num == 0:
        return None
    return s / valid_num

def accuracy_and_dice(input, target, num_classes, negative_thresh):
    input = input.unsqueeze(1)
    target = target.unsqueeze(1)

    ac = 0.
    recall = np.array([0.] * num_classes)
    total_num = np.array([0] * num_classes)
    if input.is_cuda:
        s = torch.tensor([0.]*num_classes).cuda()
    else:
        s = torch.tensor([0.]*num_classes)

    for c in zip(input, target):
        # print(c[0].size())
        # print(c[1].size())
        pred = c[0]
        gt = c[1]
        ac += metrics.accuracy_score(gt.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())
        recall += np.array([metrics.recall_score(gt.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(),
                                                 average='macro', labels=[cls]) for cls in range(num_classes)])
        if input.is_cuda:
            gt_onthot = torch.zeros(num_classes, gt.shape[1], gt.shape[2]).cuda().scatter_(0, gt, 1)
            pred_onehot = torch.zeros(num_classes, gt.shape[1], gt.shape[2]).cuda().scatter_(0, pred, 1)
        else:
            gt_onthot = torch.zeros(num_classes, gt.shape[1], gt.shape[2]).scatter_(0, gt, 1)
            pred_onehot = torch.zeros(num_classes, gt.shape[1], gt.shape[2]).scatter_(0, pred, 1)
        for i in range(num_classes):
            # have no this class
            if len(gt[gt==i]) == 0:
                continue

            s[i] += DiceCoeff().forward(pred_onehot[i].float(), gt_onthot[i].float())
            total_num[i] += 1

    acc_avg = ac / total_num[0]
    recall = recall.tolist()
    recall_avg = [0.] * num_classes
    dice_avg = [0.] * num_classes
    for i, num in enumerate(total_num):
        if num == 0:
            recall_avg[i] = -1
            dice_avg[i] = -1
        else:
            recall_avg[i] = recall[i] / num
            dice_avg[i] = s[i].item() / num

    # return dice_avg, acc_avg, recall_avg
    return {
        'dice': dice_avg,
        'acc': acc_avg,
        'recall': recall_avg,
    }

class mIoU(nn.Module):

    def __init__(self, smooth=0.01, num_classes=21):
        """
        Pytorch implement miou
        :param smooth: smooth param
        :param num_classes: number of classes
        """
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, predict, target):
        """
        Forward function
        :param predict: Variable, [bs, N]
        :param target: Variable, [bs, N]
        :return:
        """
        predict, target = predict.cpu().data, target.cpu().data
        batch_size, N = predict.size()
        eye = torch.eye(self.num_classes)
        predict_onehot = eye[predict.view(-1)]
        predict_onehot = predict_onehot.view(batch_size, N, self.num_classes).permute(2, 0, 1)  # [num_classes, bs, N]
        target_onehot = eye[target.view(-1)]
        target_onehot = target_onehot.view(batch_size, N, self.num_classes).permute(2, 0, 1)  # [num_classes, bs, N]

        ious = []
        for cls in range(1, self.num_classes):
            predict = predict_onehot[cls]  # [bs, N]
            target = target_onehot[cls]  # [bs, N]
            predict = predict.float().contiguous().view((batch_size, -1))
            target = target.float().contiguous().view((batch_size, -1))
            tp = (predict * target).sum(1)
            iou = (tp + self.smooth) / (predict.sum(1) + target.sum(1) - tp + self.smooth)
            ious.append(iou)
        return torch.stack(ious, dim=1).mean()


def mIoU_numpy(predicts, targets, num_classes):

    def jaccard(a, b, smooth=1e-3):
        """

        :param a: [H, W, n_class]
        :param b: [H, W, n_class]
        :param smooth: smooth parameter
        :return: np.array - [n_class]
        """
        assert a.shape == b.shape, 'mask1 and mask2 must have the same shape'
        assert len(a.shape) == 3

        inter = (a * b).sum(axis=(0, 1))  # (n_class)
        union = a.sum(axis=(0, 1)) + b.sum(axis=(0, 1)) - inter  # (n_class)
        iou = (inter + smooth) / (union + smooth)
        return iou

    def onehot(mask):
        H, W = mask.shape
        ret = np.zeros((H * W, num_classes))
        ret[range(H * W), mask.flatten()] = 1
        return ret.reshape(H, W, num_classes)

    num_images = len(predicts)
    iou = []
    for pred, target in zip(predicts, targets):
        pred = onehot(pred)  # H, W, nclass
        target = onehot(target)  # H, W, nclass
        iou.append(jaccard(pred, target))  # (nclass)

    iou = np.stack(iou, axis=1)  # (nclass, n_images)

    cls_iou = []  # miou for each class
    for cls in range(1, num_classes):
        count = np.zeros(num_images)  # (n_images)
        thresh = np.arange(0.5, 1., 0.05)
        for t in thresh:
            match = iou[cls] > t # (n_images)
            count += match
        count /= len(thresh)
        cls_iou.append(count.mean())
    return cls_iou
