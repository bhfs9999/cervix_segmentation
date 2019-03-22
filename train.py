import os
import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from sklearn import metrics
from datetime import datetime
from optparse import OptionParser
from torch.utils.data import DataLoader

from nets.model_resnet34_bn import unet34
from utils.comparator import Comparator
from dataset.Cervix import Cervix
from utils.summary import LogSummary
from utils.init_function import weights_init_xavier
from utils.data_augmentation import TrainAugmentation, TestNormalization, get_train_augmentation, get_valid_normalization
from utils.utils import AverageMeter, save_model, load_model

auto_op_mapping = {'loss': 'less_than', 'acc': 'larger_than', 'dice': 'larger_than', 'none': 'equal'}

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def valid_net(net, valid_dataloader, valid_strategy, tensor_writer):
    net.set_mode('valid')
    valid_iter = iter(valid_dataloader)
    valid_interval = len(valid_dataloader)
    acc_counter, loss_counter = AverageMeter(), AverageMeter()
    recall_counter = [AverageMeter() for _ in range(valid_strategy['num_categories'])]
    dice_counter = [AverageMeter() for _ in range(valid_strategy['num_categories'])]
    valid_target_list, valid_predict_list = [], []
    valid_target_per_img_list, valid_predict_per_img_list = [], []
    for test_step in range(valid_interval):
        # Data
        imgs, true_masks = next(valid_iter)
        imgs = imgs.cuda()
        true_masks = true_masks.cuda()

        with torch.no_grad():
            # forward
            masks_probs = net(imgs)

            valid_predict_list.extend([pred.astype(np.uint8) for pred in
                                       (masks_probs.squeeze() > valid_strategy['mask_threshold']).cpu().numpy()])
            valid_target_list.extend([truth.astype(np.uint8) for truth in true_masks.cpu().numpy()])

            valid_predict_per_img_list.extend([pred.astype(np.uint8) for pred in (
                        (masks_probs.view(masks_probs.size(0), -1) > 0).sum(dim=1) > train_strategy[
                    'neg_thresh']).cpu().numpy()])
            valid_target_per_img_list.extend([truth.astype(np.uint8) for truth in
                                              (true_masks.view(true_masks.size(0), -1).max(dim=1)[0]).cpu().numpy()])

            loss = net.criterion(masks_probs, true_masks)

            # dice, acc = net.metric(masks_probs > valid_strategy['mask_threshold'], true_masks)
            result = net.metric(masks_probs, true_masks, valid_strategy['num_categories'], valid_strategy['mask_threshold'])
            dice = [0 if x==-1 else x for x in result['dice']]
            acc = result['acc']
            recall = [0 if x==-1 else x for x in result['recall']]

            acc_counter.update(acc, valid_strategy['batch_size'])
            loss_counter.update(loss.item(), valid_strategy['batch_size'])
            for i, counter in enumerate(dice_counter):
                if dice[i] is not None:
                    counter.update(dice[i], train_strategy['batch_size'])
            for i, counter in enumerate(recall_counter):
                if recall[i] is not None:
                    recall_counter[i].update(recall[i], valid_strategy['batch_size'])
    acc_per_img = metrics.accuracy_score(valid_target_per_img_list, valid_predict_per_img_list)
    recall_per_img = [
        metrics.recall_score(valid_target_per_img_list, valid_predict_per_img_list, average='macro', labels=[cls]) for
        cls in range(valid_strategy['num_categories'])]

    metric_names = []
    scalars = []

    metric_names += ['acc']
    scalars += [acc_counter.avg]
    metric_names += ['acc_per_img']
    scalars += [acc_per_img]
    metric_names += ['dice-{}'.format(cls) for cls in range(train_strategy['num_categories'])]
    scalars += [counter.avg for counter in dice_counter]
    metric_names += ['recall-{}'.format(cls) for cls in range(valid_strategy['num_categories'])]
    scalars += [counter.avg for counter in recall_counter]
    metric_names += ['recall_per_img-{}'.format(cls) for cls in range(valid_strategy['num_categories'])]
    scalars += [recall_per_img[i] for i in range(len(recall_per_img))]
    metric_names += ['loss']
    scalars += [loss_counter.avg]
    tensor_writer.write_scalars(scalars, metric_names, valid_strategy['step'], tag='valid')
    print('='*20)
    print('Validation: || Loss: %.4f || acc: %.4f || dice: %s' % \
              (loss_counter.avg, acc_counter.avg,
               ', '.join(['{:.4f}'.format(x.avg) for x in dice_counter])))
    print('='*20)
    return {
        'acc': acc_counter.avg,
        'dice': [counter.avg for counter in dice_counter],
        'loss': loss_counter.avg,
        'recall': [counter.avg for counter in recall_counter]
    }

def train_net(net, dataloaders, train_strategy):
    # Data_parallel placeholder
    parallel_tag = False

    dir_checkpoint = './checkpoints/{}/{}/{}/'.format(train_strategy['set_name'], train_strategy['data_type'], train_strategy['tag'])
    dir_tensorboard = './tensorboardX/{}/{}/{}/'.format(train_strategy['set_name'], train_strategy['data_type'], datetime.now().strftime('%b%d_%H-%M-%S_') + train_strategy['tag'])

    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    if not os.path.exists(dir_tensorboard):
        os.makedirs(dir_tensorboard)

    learning_rate = train_strategy['lr']

    tensor_writer = LogSummary(dir_tensorboard)

    optimizer = optim.SGD(net.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=0.0005)

    train_loader = dataloaders[0]
    valid_loader = dataloaders[1]

    start_step = 0
    end_step = train_strategy['epochs'] * len(train_loader)

    lr_adaptiver = Comparator(learning_rate, train_strategy['auto'], auto_op_mapping[train_strategy['auto_metric']],
                              (np.array(train_strategy['lr_strategy']) * len(train_loader)).tolist(),
                              decay_ratio=train_strategy['lr_decay'], max_decay=7)
    lr_indication = None

    train_dataloader = train_loader
    train_iter = iter(train_dataloader)
    valid_dataloader = valid_loader

    train_interval = 50
    log_interval = len(train_dataloader)
    # eval_interval = 10
    eval_interval = len(train_dataloader)

    best_valid_dice = 0.
    epoch_batch_count = 0

    train_predict_list = []
    train_target_list = []

    train_predict_per_img_list = []
    train_target_per_img_list = []

    acc_counter, loss_counter = AverageMeter(), AverageMeter()
    recall_counter = [AverageMeter() for _ in range(train_strategy['num_categories'])]
    dice_counter = [AverageMeter() for _ in range(train_strategy['num_categories'])]

    for step in range(start_step, end_step + 1):
        net.set_mode('train')
        # update lr if no auto decay
        if not train_strategy['auto']:
            lr_indication = lr_adaptiver(step)
            if lr_indication == 'decay':
                learning_rate = lr_adaptiver.lr
                adjust_learning_rate(optimizer, learning_rate)
            elif lr_indication == 'invariable':
                pass
            else:
                raise ValueError

        # get data from data_loader
        imgs, true_masks = next(train_iter)
        epoch_batch_count += 1
        imgs = imgs.cuda()
        true_masks = true_masks.cuda()

        # forward
        t0 = time.time()
        masks_probs = net(imgs)

        train_predict_list.extend([pred.astype(np.uint8) for pred in (masks_probs.squeeze() > train_strategy['mask_threshold']).cpu().numpy()])
        train_target_list.extend([truth.astype(np.uint8) for truth in true_masks.cpu().numpy()])

        train_predict_per_img_list.extend([pred.astype(np.uint8) for pred in ((masks_probs.view(masks_probs.size(0), -1) > 0).sum(dim=1) > train_strategy['neg_thresh']).cpu().numpy()])
        train_target_per_img_list.extend([truth.astype(np.uint8) for truth in (true_masks.view(true_masks.size(0), -1).max(dim=1)[0]).cpu().numpy()])

        loss = net.criterion(masks_probs, true_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()

        # dice, acc = net.metric(masks_probs > train_strategy['mask_threshold'], true_masks)
        # dice, acc, recall = net.metric(masks_probs, true_masks, train_strategy['num_categories'], train_strategy['mask_threshold'])
        result = net.metric(masks_probs, true_masks, train_strategy['num_categories'], train_strategy['mask_threshold'])
        dice = result['dice']
        acc = result['acc']
        recall = result['recall']

        acc_counter.update(acc, train_strategy['batch_size'])
        loss_counter.update(loss.item(), train_strategy['batch_size'])
        for i, counter in enumerate(dice_counter):
            if dice[i] is not None:
                counter.update(dice[i], train_strategy['batch_size'])
        for i, counter in enumerate(recall_counter):
            if recall[i] is not None:
                counter.update(recall[i], train_strategy['batch_size'])

        if step % train_interval == 0:
            pstring = 'timer: %.4f sec.' % (t1 - t0)
            print(pstring)
            pstring = 'iter ' + repr(step) + ' || Loss: %.4f || lr: %.5f || acc: %.4f || dice: %s' % \
                      (loss.item(), learning_rate, acc, ', '.join(['{:.4f}'.format(x) for x in dice]))
            print(pstring)

        # reset iter
        if epoch_batch_count >= len(train_dataloader):
            train_iter = iter(train_dataloader)
            epoch_batch_count = 0

        if step % log_interval == 0 and step != start_step:
            acc_per_img = metrics.accuracy_score(train_target_per_img_list, train_predict_per_img_list)
            recall_per_img = [metrics.recall_score(train_target_per_img_list, train_predict_per_img_list, average='macro', labels=[cls]) for cls in range(train_strategy['num_categories'])]

            metric_names = []
            scalars = []

            metric_names += ['acc']
            scalars += [acc_counter.avg]
            metric_names += ['acc_per_img']
            scalars += [acc_per_img]
            metric_names += ['dice-{}'.format(cls) for cls in range(train_strategy['num_categories'])]
            scalars += [counter.avg for counter in dice_counter]
            metric_names += ['recall-{}'.format(cls) for cls in range(train_strategy['num_categories'])]
            scalars += [counter.avg for counter in recall_counter]
            metric_names += ['recall_per_img-{}'.format(cls) for cls in range(train_strategy['num_categories'])]
            scalars += [recall_per_img[i] for i in range(len(recall_per_img))]
            metric_names += ['loss']
            scalars += [loss_counter.avg]
            metric_names += ['learning_rate']
            scalars += [learning_rate]
            tensor_writer.write_scalars(scalars, metric_names, step, tag='train')
            print('=' * 20)
            print('Epoch %d ending, avg result: || Loss: %.4f || acc: %.4f || dice: %s' % \
                  (step // len(train_dataloader), loss_counter.avg, acc_counter.avg,
                   ', '.join(['{:.4f}'.format(x.avg) for x in dice_counter])))

            # reset counters
            acc_counter, loss_counter = AverageMeter(), AverageMeter()
            recall_counter = [AverageMeter() for _ in range(train_strategy['num_categories'])]
            dice_counter = [AverageMeter() for _ in range(train_strategy['num_categories'])]
            train_predict_list, train_target_list = [], []

        if step % eval_interval == 0 and step != start_step:
            valid_strategy = {
                'step': step,
                'batch_size': train_strategy['batch_size'],
                'mask_threshold': train_strategy['mask_threshold'],
                'num_categories': train_strategy['num_categories']
            }
            valid_result = valid_net(net, valid_dataloader, valid_strategy, tensor_writer)

            # todo more generate
            valid_dice = valid_result['dice'][2]
            # save best checkpoint, the metric is weighted f1 score
            if valid_dice > best_valid_dice:
                best_valid_dice = valid_dice
                print('Saving best valid dice state, iter:', step)
                save_model(net, optimizer, dir_checkpoint, 'best_valid_dice_checkpoint.pth', step, learning_rate, parallel_tag)

            if train_strategy['auto']:
                lr_indication = lr_adaptiver(valid_result[train_strategy['auto_metric']])
                if lr_indication == 'decay':
                    learning_rate = lr_adaptiver.lr
                    adjust_learning_rate(optimizer, learning_rate)
                elif lr_indication == 'early_stop':
                    print('Early stop!')
                    sys.exit()
                elif lr_indication == 'invariable':
                    pass
                else:
                    raise ValueError
        # save_model
        if step % (1 * len(train_dataloader)) == 0 and step != start_step:
            print('Saving last state, iter:', step)
            save_model(net, optimizer, dir_checkpoint, 'last_checkpoint.pth', step, learning_rate, parallel_tag)

        # save_model regularly
        if step % (train_strategy['regularly_save'] * len(train_dataloader)) == 0 and step != start_step:
            print('Saving state regularly, iter:', step)
            save_model(net, optimizer, dir_checkpoint, '{}_checkpoint.pth'.format(step), step, learning_rate, parallel_tag)


def get_args():
    parser = argparse.ArgumentParser()
    # Training setting
    parser.add_argument('--train_set', dest='train_set',
                        type=str, default='train',
                        choices=['train', 'train_pos'],
                        help='which train data set to use')
    parser.add_argument('--valid_set', dest='valid_set',
                        type=str, default='valid',
                        choices=['valid', 'valid_pos'],
                        help='which valid data set to use')
    parser.add_argument('--epochs', dest='epochs', type=int,
                      help='number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=8,
                      type=int, help='batch size')
    parser.add_argument('--lr', dest='lr', default=0.001,
                      type=float, help='learning rate')
    parser.add_argument('--resume', dest='resume',
                      default=None, help='model dir to resume')
    parser.add_argument('--num_classes', dest='num_classes', type=int,
                      default=1, help='num classes of classification')
    parser.add_argument('--num_categories', dest='num_categories', type=int,
                      default=2, help='num classes of classification')
    parser.add_argument('--schedule', type=int, nargs='+', default=[90, 150],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr_decay', type=float, dest='lr_decay', default=0.1,
                        help='lr decay ratio')
    parser.add_argument('--auto', dest='auto', action='store_true',
                        help='whether to use auto learning decay')
    parser.add_argument('--auto_metric', dest='auto_metric',
                        help='when use "auto", select a metric as every epoch s judgement',
                        default='none', choices=['none', 'loss', 'acc', 'dice', 'mIoU'])
    parser.add_argument('--loss', type=str, default='cross_entropy', dest='loss',
                        choices=['BCE', 'cross_entropy', 'Lovasz_Softmax', 'Lovasz_sigmoid', 'Focal_loss', 'dice_loss'],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--mask_threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--activation', dest='activation', type=str, default='sigmoid',
                        choices=['softmax', 'sigmoid'],
                        help='Which activation function to use')
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=10,
                        help='Interval between two saving checkpoints')
    parser.add_argument('--refine', dest='refine', type=str, default=None,
                        help='Whether to use refine strategy')
    parser.add_argument('--neg_thresh', dest='neg_thresh', default=1000, type=int,
                        help='When calculate recall per image, use to set the negative prediction thresh-hold')
    # default:[1, 20]
    parser.add_argument('--class_weights', dest='class_weights', type=int, nargs='+', default=None,
                        help='Loss weights which are used to optimize the less class')
    # Data setting
    parser.add_argument('--scale', dest='scale', type=float,
                      default=512, help='input target size')
    parser.add_argument('--data_type', dest='data_type', type=str, choices=['acid', 'iodine'],
                      default=None, help='input data type')
    # Gpu setting
    parser.add_argument('--gpus', dest='gpus', default=None, nargs='+', type=int,
                        help='When use dataparallel, it is needed to set gpu Ids')
    # Other setting
    parser.add_argument('--tag', type=str, default=None, dest='tag',
                      help='Used to distinguish different experiments')


    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    data_set_root = '/home/mxj/data/Cervix/Segmentation/cervix_resize_600_segmentation'
    # data_set_root = '/home/mxj/data/Cervix/Segmentation/second_batch'
    if args.class_weights is not None:
        class_weights = torch.FloatTensor(args.class_weights).cuda()
    else:
        class_weights = None
    net = unet34(num_classes=args.num_classes, criterion=args.loss, activation=args.activation, init_function=weights_init_xavier, class_weights=class_weights)

    args.load = '/home/mxj/data/CommonWeights/resnet34-333f7ec4.pth'
    # args.refine = './checkpoints/acid/Focal_loss_1_20_transfer_run1/best_valid_dice_checkpoint.pth'
    # assert args.data_type in args.refine

    experiment_tag = ''
    experiment_tag += args.loss
    if args.auto:
        experiment_tag += '_'.join(['', 'auto', args.auto_metric])
    if args.class_weights is not None:
        experiment_tag += '_'.join([''] + [str(x) for x in args.class_weights])
    if args.refine is not None:
        # load_model(net, args.refine, key_word='logit')
        load_model(net, args.refine)
        experiment_tag += '_refine'
        print('Model start refining from {}'.format(args.refine))
    else:
        if args.load is not None:
            net.load_pretrain(args.load)
            experiment_tag += '_transfer'
            print('Model loaded from {}'.format(args.load))
    experiment_tag += '_{}'.format(args.tag)

    net.cuda()
    if args.gpus is not None:
        assert isinstance(args.gpus, list) and len(args.gpus) > 1
        net = torch.nn.DataParallel(net, device_ids=args.gpus)
        parallel_tag = True
    # cudnn.benchmark = True # faster convolutions, but more memory

    train_dataset = Cervix(root=data_set_root, data_set_type=args.train_set, transform=get_train_augmentation, data_type=args.data_type, tf_learning=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  collate_fn=train_dataset.classify_collate, drop_last=True)
    valid_dataset = Cervix(root=data_set_root, data_set_type=args.valid_set, transform=get_valid_normalization, data_type=args.data_type, tf_learning=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                  collate_fn=valid_dataset.classify_collate, drop_last=False)

    train_strategy= {
        'epochs': args.epochs,
        'img_scale': args.scale,
        'num_classes': args.num_classes,
        'num_categories': args.num_categories,
        'lr_strategy': args.schedule,
        'loss_type': args.loss,
        'tag': experiment_tag,
        'lr': args.lr,
        'mask_threshold': args.mask_threshold,
        'activation': args.activation,
        'power': 0.9,
        'gpus': args.gpus,
        'auto': args.auto,
        'auto_metric': args.auto_metric,
        'lr_decay': args.lr_decay,
        'batch_size': args.batch_size,
        'regularly_save': args.save_interval,
        'data_type': args.data_type,
        'neg_thresh': args.neg_thresh,
        'set_name': data_set_root.split('/')[-1]
    }

    try:
        train_net(net, [train_dataloader, valid_dataloader], train_strategy)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
