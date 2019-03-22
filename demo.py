import os
import sys
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from nets.model_resnet34_bn import unet34
from dataset.Cervix import Cervix
from utils.data_augmentation import get_valid_normalization
from sklearn import metrics

label_map = np.array([[0, 0, 0], [84, 255, 159], [100, 149, 237], [255, 187, 255]])
# mask_mapping = np.array([0, 1, 1, 1])
mask_mapping = np.array([0, 1, 1, 0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', dest='set',
                        type=str, default='demo',
                        choices=['train', 'valid', 'test', 'demo'],
                        help='which data set to use')
    parser.add_argument('--batch_size', dest='batch_size', default=1,
                      type=int, help='batch size')
    parser.add_argument('--mask_threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=512)
    # choices=['dice', 'cross_entropy']
    parser.add_argument('--loss', type=str, default='cross_entropy', dest='loss',
                        choices=['BCE', 'cross_entropy', 'Lovasz_Softmax', 'Lovasz_sigmoid', 'Focal_loss', 'dice_loss'],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('-n', '--num_classes', dest='num_classes', type=int,
                      default=1, help='num classes of classification')
    parser.add_argument('--num_categories', dest='num_categories', type=int,
                      default=2, help='num classes of classification')
    parser.add_argument('--activation', dest='activation', type=str,
                        default='sigmoid', choices=['sigmoid', 'softmax'],
                        help='Which activation function to use')
    parser.add_argument('--mask_type', type=str, default='Binary', dest='mask_type',
                      help='Used to distinguish different experiments')
    parser.add_argument('--draw', dest='draw', action='store_true', default=True,
                        help='Whether to save prediction result')
    parser.add_argument('--no_gt', dest='no_gt', action='store_true', default=False,
                        help='Whether to use gt to get metrics')
    parser.add_argument('--model', dest='model', type=str, default='best_valid',
                        help='the checkpoint name which to be loaded')
    parser.add_argument('--phase', dest='phase', action='store_true', default=False,
                        help='Whether to save phase1 result')
    parser.add_argument('--detection', dest='detection', default=False, action='store_true',
                        help='whether to use detection region')
    parser.add_argument('--neg_thresh', dest='neg_thresh', default=0, type=int,
                        help='When calculate recall per image, use to set the negative prediction thresh-hold')
    # gpu setting
    parser.add_argument('--gpus', dest='gpus', default=None, nargs='+', type=int,
                        help='When use dataparallel, it is needed to set gpu Ids')

    return parser.parse_args()


def predict_img(net, img, mask, test_strategy, meta_info):
    true_mask = mask

    with torch.no_grad():
        img = img.cuda()
        if mask is not None:
            true_mask = true_mask.cuda()
        mask_prob = net(img)

    if mask is not None:
        if test_strategy['num_classes'] == 1 and test_strategy['num_categories']:
            dice_metrics = []
        else:
            dice_metrics = []
        # dice, acc, recall = net.metric(mask_prob, true_mask, test_strategy['num_categories'], test_strategy['mask_threshold'], test_strategy['neg_thresh'])
        result = net.metric(mask_prob, true_mask, test_strategy['num_categories'], test_strategy['mask_threshold'], test_strategy['neg_thresh'])
    if test_strategy['batch_size'] > 1:
        return None, result
    else:
        ori_h, ori_w = meta_info['ori_size']
        mask_np = cv2.resize(mask_prob.squeeze(0).cpu().numpy().transpose((1, 2, 0)), dsize=(ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        if len(mask_np.shape) == 2:
            mask_np = mask_np[:, :, np.newaxis]
        mask_np = mask_np.transpose((2, 0, 1))

        if mask_np.shape[0] == 1:
            mask_np = (mask_np > test_strategy['mask_threshold']).astype(np.uint8).squeeze(0)
        else:
            mask_np = np.argmax(mask_np, axis=0).reshape(ori_h, ori_w)

        if mask is not None:
            return mask_np, result
        else:
            return mask_np, None

def mask_to_image(mask):
    return Image.fromarray((mask).astype(np.uint8))

def save_mask_and_gt(ori_img, mask_gt, mask, output_path, dice):
    '''

    :param mask_gt: origin mask ground truth
    :param mask: origin sized predicted mask
    :param output_path: the path which is to be saved
    :param dice: seg dice
    '''
    # final_mask = label_map[mask]
    # true_mask = label_map[mask_gt]
    gt_image = ori_img.copy()
    pred_image = ori_img.copy()
    mask_gt = mask_gt.astype(np.uint8)
    mask = mask.astype(np.uint8)

    ori_h, ori_w = mask_gt.shape[:2]
    final_output = np.zeros((ori_h, 2 * ori_w + 10, 3), dtype=np.uint8)

    _, thresh = cv2.threshold(mask_gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    cv2.drawContours(gt_image, contours, -1, (84, 255, 159), 2)
    # cnt = contours[0]
    # approx = cv2.approxPolyDP(cnt, 3, True)
    # cv2.polylines(gt_image, [approx], True, (84, 255, 159), 2)
    final_output[:, 0:ori_w, :] = gt_image

    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    cv2.drawContours(pred_image, contours, -1, (84, 255, 159), 2)

    # # darw dice
    # cv2.putText(pred_image, str(dice), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (84, 255, 159), 2)

    # cnt = contours[0]
    # approx = cv2.approxPolyDP(cnt, 3, True)
    # cv2.polylines(pred_image, [approx], True, (84, 255, 159), 2)
    final_output[:, ori_w + 10:, :] = pred_image
    final_output[:, ori_w:ori_w + 10, :] = np.array([255, 255, 255])

    result = mask_to_image(final_output)
    result.save(output_path)
    print("Mask example saved to {}".format(output_path))


if __name__ == "__main__":
    args = get_args()

    generate_tag = False


    # acid
    args.data_type = 'acid'
    data_set_root = '/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation'
    model_file = '/data/mxj/project/Cervix/Cervix_segmentation/checkpoints/' \
                 'second_batch/acid/Focal_loss_1_20_transfer_neg_InNeg_run1/best_valid_dice_checkpoint.pth'
    output_path = '/data/lxc/output/cervix_seg/demo'

    net = unet34(num_classes=args.num_classes, criterion=args.loss, activation=args.activation)

    if args.draw:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    print("Loading model {}".format(model_file))
    pre_model = torch.load(model_file)['state_dict']
    pre_model_dict = {k: v for k, v in pre_model.items() if ('criterion_loss' not in k)}
    net.load_state_dict(pre_model_dict)
    print("Model loaded !")

    net.cuda()
    net.eval()
    if args.gpus is not None:
        assert isinstance(args.gpus, list) and len(args.gpus) > 1
        net = torch.nn.DataParallel(net, device_ids=args.gpus)
        parallel_tag = True

    test_dataset = Cervix(root=data_set_root, data_set_type=args.set, transform=get_valid_normalization,
                          data_type=args.data_type, tf_learning=True, test_mode=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                  collate_fn=test_dataset.classify_collate, drop_last=False)

    test_strategy = {
        'num_classes': args.num_classes,
        'num_categories': args.num_categories,
        'loss_type': args.loss,
        'scale_factor': args.scale,
        'out_threshold': args.mask_threshold,
        'mask_threshold': args.mask_threshold,
        'activation': args.activation,
        'draw': args.draw,
        'batch_size': args.batch_size,
        'neg_thresh': args.neg_thresh
    }

    if test_strategy['num_classes'] == 1 and test_strategy['num_categories'] == 2:
        dice_metrics = [0]
        dice_valid_mask = [len(test_dataloader)]
        recall_metrics = [0] * 2
        recall_valid_mask = [len(test_dataloader)] * 2
    else:
        dice_metrics = [0] * test_strategy['num_classes']
        dice_valid_mask = [len(test_dataloader)] * test_strategy['num_classes']
        recall_metrics = [0] * test_strategy['num_classes']
        recall_valid_mask = [len(test_dataloader)] * test_strategy['num_classes']

    dice_metrics = np.array(dice_metrics, dtype=np.float32)
    recall_metrics = np.array(recall_metrics, dtype=np.float32)
    test_predict_per_img_list = []
    test_target_per_img_list = []

    dice_info = {}
    for i, batch in enumerate(test_dataloader):
        img = batch[0]
        if not args.no_gt:
            true_mask = batch[1]
            meta_info = batch[2]
            test_target_per_img_list.extend([truth.astype(np.uint8) for truth in
                                             (true_mask.view(true_mask.size(0), -1).max(dim=1)[0]).cpu().numpy()])
        else:
            true_mask = None
            meta_info = batch[1]
        print("Predicting image {}/{} ...".format(i+1, len(test_dataloader)))
        img_name = meta_info[0]['img_path'].split('/')[-1]
        if args.batch_size == 1:
            meta_info = meta_info[0]
        mask, result = predict_img(net=net,
                           img=img,
                           mask=true_mask,
                           test_strategy=test_strategy,
                           meta_info=meta_info)
        dice_info[img_name] = result['dice']
        dice_list = [result['dice']]
        acc = result['acc']
        recall_list = result['recall']
        # test_predict_per_img_list.extend([pred.astype(np.uint8) for pred in ((mask.reshape(-1) > 0).sum() > test_strategy['neg_thresh'])])
        test_predict_per_img_list.extend([(mask.reshape(-1) > 0).sum() > test_strategy['neg_thresh']])

        if args.draw:
            ori_img_name = meta_info['img_path'].split('/')[-1].strip('.jpg')
            out_file_parts = []
            out_file_parts.append('{}_out'.format(ori_img_name))
            out_file_parts.append('dice')
            for i, item in enumerate(dice_list):
                if item is None or item == -1:
                    out_file_parts.append('[{}]_{}'.format(i, 'None'))
                else:
                    # tmp_item = item.item()
                    out_file_parts.append('[{}]_{:.2f}'.format(i, item))
            out_file_parts.append('recall')
            for i, item in enumerate(recall_list):
                if item is None or item == -1:
                    out_file_parts.append('[{}]_{}'.format(i, 'None'))
                else:
                    # tmp_item = item.item()
                    out_file_parts.append('[{}]_{:.2f}'.format(i, item))
            out_file_name = '_'.join(out_file_parts) + '.png'
            out_file = os.path.join(output_path, out_file_name)

        for i, item in enumerate(dice_list):
            if item is None or item == -1:
                dice_valid_mask[i] -= 1
            else:
                dice_metrics[i] += item

        for i, item in enumerate(recall_list):
            if item is None or item == -1:
                recall_valid_mask[i] -= 1
            else:
                recall_metrics[i] += item

            mask_gt = Image.open(meta_info['mask_path'])
            # if meta_info[0]['bbox'] is not None:
            #     mask_gt = mask_gt.crop(meta_info[0]['bbox'])
            mask_gt = np.array(mask_gt, dtype=np.int64)

        if args.draw:
            if args.mask_type == 'Binary':
                mask_gt = mask_mapping[mask_gt]
            elif args.mask_type == 'Real':
                pass
            else:
                raise ValueError
            ori_img = Image.open(meta_info['img_path'])
            ori_img = np.array(ori_img, dtype=np.uint8)
            save_mask_and_gt(ori_img, mask_gt, mask, out_file, result['dice'])

    dice_metrics /= np.array(dice_valid_mask, dtype=np.float32)
    recall_metrics /= np.array(recall_valid_mask, dtype=np.float32)
    acc_per_img = metrics.accuracy_score(test_target_per_img_list, test_predict_per_img_list)
    recall_per_img = [metrics.recall_score(test_target_per_img_list, test_predict_per_img_list, average='macro', labels=[cls]) for
        cls in range(test_strategy['num_categories'])]
    precision_per_img = [metrics.precision_score(test_target_per_img_list, test_predict_per_img_list, average='macro', labels=[cls]) for
        cls in range(test_strategy['num_categories'])]
    print_list = []
    for i, item in enumerate(dice_metrics):
        print_list.append('dice[{}]_{:.4f}'.format(i, item))
    for i, item in enumerate(recall_metrics):
        print_list.append('recall[{}]_{:.4f}'.format(i, item))
    for i, item in enumerate(recall_per_img):
        print_list.append('recall_per_img[{}]_{:.4f}'.format(i, item))
    for i, item in enumerate(precision_per_img):
        print_list.append('precision_per_img[{}]_{:.4f}'.format(i, item))
    print_list.append('acc_{:.4f}'.format(acc_per_img))
    final_log = 'Test over: ' + ' '.join(print_list)
    print(final_log)



    # iodine
    args.data_type = 'iodine'
    data_set_root = '/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation'
    model_file = '/data/mxj/project/Cervix/Cervix_segmentation/checkpoints/' \
                 'second_batch/iodine/Focal_loss_1_20_transfer_neg_InNeg_run1/best_valid_dice_checkpoint.pth'
    output_path = '/data/lxc/output/cervix_seg/demo'

    net = unet34(num_classes=args.num_classes, criterion=args.loss, activation=args.activation)

    if args.draw:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    print("Loading model {}".format(model_file))
    pre_model = torch.load(model_file)['state_dict']
    pre_model_dict = {k: v for k, v in pre_model.items() if ('criterion_loss' not in k)}
    net.load_state_dict(pre_model_dict)
    print("Model loaded !")

    net.cuda()
    net.eval()
    if args.gpus is not None:
        assert isinstance(args.gpus, list) and len(args.gpus) > 1
        net = torch.nn.DataParallel(net, device_ids=args.gpus)
        parallel_tag = True

    test_dataset = Cervix(root=data_set_root, data_set_type=args.set, transform=get_valid_normalization,
                          data_type=args.data_type, tf_learning=True, test_mode=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                 collate_fn=test_dataset.classify_collate, drop_last=False)

    test_strategy = {
        'num_classes': args.num_classes,
        'num_categories': args.num_categories,
        'loss_type': args.loss,
        'scale_factor': args.scale,
        'out_threshold': args.mask_threshold,
        'mask_threshold': args.mask_threshold,
        'activation': args.activation,
        'draw': args.draw,
        'batch_size': args.batch_size,
        'neg_thresh': args.neg_thresh
    }

    if test_strategy['num_classes'] == 1 and test_strategy['num_categories'] == 2:
        dice_metrics = [0]
        dice_valid_mask = [len(test_dataloader)]
        recall_metrics = [0] * 2
        recall_valid_mask = [len(test_dataloader)] * 2
    else:
        dice_metrics = [0] * test_strategy['num_classes']
        dice_valid_mask = [len(test_dataloader)] * test_strategy['num_classes']
        recall_metrics = [0] * test_strategy['num_classes']
        recall_valid_mask = [len(test_dataloader)] * test_strategy['num_classes']

    dice_metrics = np.array(dice_metrics, dtype=np.float32)
    recall_metrics = np.array(recall_metrics, dtype=np.float32)
    test_predict_per_img_list = []
    test_target_per_img_list = []

    dice_info = {}
    for i, batch in enumerate(test_dataloader):
        img = batch[0]
        if not args.no_gt:
            true_mask = batch[1]
            meta_info = batch[2]
            test_target_per_img_list.extend([truth.astype(np.uint8) for truth in
                                             (true_mask.view(true_mask.size(0), -1).max(dim=1)[0]).cpu().numpy()])
        else:
            true_mask = None
            meta_info = batch[1]
        print("Predicting image {}/{} ...".format(i + 1, len(test_dataloader)))
        img_name = meta_info[0]['img_path'].split('/')[-1]
        if args.batch_size == 1:
            meta_info = meta_info[0]
        mask, result = predict_img(net=net,
                                   img=img,
                                   mask=true_mask,
                                   test_strategy=test_strategy,
                                   meta_info=meta_info)
        dice_info[img_name] = result['dice']
        dice_list = [result['dice']]
        acc = result['acc']
        recall_list = result['recall']
        # test_predict_per_img_list.extend([pred.astype(np.uint8) for pred in ((mask.reshape(-1) > 0).sum() > test_strategy['neg_thresh'])])
        test_predict_per_img_list.extend([(mask.reshape(-1) > 0).sum() > test_strategy['neg_thresh']])

        if args.draw:
            ori_img_name = meta_info['img_path'].split('/')[-1].strip('.jpg')
            out_file_parts = []
            out_file_parts.append('{}_out'.format(ori_img_name))
            out_file_parts.append('dice')
            for i, item in enumerate(dice_list):
                if item is None or item == -1:
                    out_file_parts.append('[{}]_{}'.format(i, 'None'))
                else:
                    # tmp_item = item.item()
                    out_file_parts.append('[{}]_{:.2f}'.format(i, item))
            out_file_parts.append('recall')
            for i, item in enumerate(recall_list):
                if item is None or item == -1:
                    out_file_parts.append('[{}]_{}'.format(i, 'None'))
                else:
                    # tmp_item = item.item()
                    out_file_parts.append('[{}]_{:.2f}'.format(i, item))
            out_file_name = '_'.join(out_file_parts) + '.png'
            out_file = os.path.join(output_path, out_file_name)

        for i, item in enumerate(dice_list):
            if item is None or item == -1:
                dice_valid_mask[i] -= 1
            else:
                dice_metrics[i] += item

        for i, item in enumerate(recall_list):
            if item is None or item == -1:
                recall_valid_mask[i] -= 1
            else:
                recall_metrics[i] += item

            mask_gt = Image.open(meta_info['mask_path'])
            # if meta_info[0]['bbox'] is not None:
            #     mask_gt = mask_gt.crop(meta_info[0]['bbox'])
            mask_gt = np.array(mask_gt, dtype=np.int64)

        if args.draw:
            if args.mask_type == 'Binary':
                mask_gt = mask_mapping[mask_gt]
            elif args.mask_type == 'Real':
                pass
            else:
                raise ValueError
            ori_img = Image.open(meta_info['img_path'])
            ori_img = np.array(ori_img, dtype=np.uint8)
            save_mask_and_gt(ori_img, mask_gt, mask, out_file, result['dice'])

    dice_metrics /= np.array(dice_valid_mask, dtype=np.float32)
    recall_metrics /= np.array(recall_valid_mask, dtype=np.float32)
    acc_per_img = metrics.accuracy_score(test_target_per_img_list, test_predict_per_img_list)
    recall_per_img = [
        metrics.recall_score(test_target_per_img_list, test_predict_per_img_list, average='macro', labels=[cls]) for
        cls in range(test_strategy['num_categories'])]
    precision_per_img = [
        metrics.precision_score(test_target_per_img_list, test_predict_per_img_list, average='macro', labels=[cls]) for
        cls in range(test_strategy['num_categories'])]
    print_list = []
    for i, item in enumerate(dice_metrics):
        print_list.append('dice[{}]_{:.4f}'.format(i, item))
    for i, item in enumerate(recall_metrics):
        print_list.append('recall[{}]_{:.4f}'.format(i, item))
    for i, item in enumerate(recall_per_img):
        print_list.append('recall_per_img[{}]_{:.4f}'.format(i, item))
    for i, item in enumerate(precision_per_img):
        print_list.append('precision_per_img[{}]_{:.4f}'.format(i, item))
    print_list.append('acc_{:.4f}'.format(acc_per_img))
    final_log = 'Test over: ' + ' '.join(print_list)
    print(final_log)

    # dice_sort = sorted(dice_info.items(), key=lambda x: x[1], reverse=True)
    # for name, dice in dice_sort[:20]:
    #     print(name, dice)


