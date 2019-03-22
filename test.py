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
mask_mapping = np.array([0, 1, 2, 0])

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', dest='test_set',
                        type=str, default='train',
                        choices=['test', 'test_pos'],
                        help='which train data set to use')
    parser.add_argument('--batch_size', dest='batch_size', default=1,
                      type=int, help='batch size')
    parser.add_argument('--mask_threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=512)
    # choices=['acid', 'iodine']
    parser.add_argument('--data_type', type=str, default='acid', dest='data_type',
                        help='What type of data used to train')
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
    parser.add_argument('--draw', dest='draw', action='store_true', default=False,
                        help='Whether to save prediction result')
    parser.add_argument('--save_phase', dest='save_phase', action='store_true', default=False,
                        help='Whether to save phase1 result')
    parser.add_argument('--save_mask', dest='save_mask', action='store_true', default=False,
                        help='Whether to save mask in phase1 to assist classification')
    parser.add_argument('--no_gt', dest='no_gt', action='store_true', default=False,
                        help='Whether to use gt to get metrics')
    parser.add_argument('--model', dest='model', type=str, default='best_valid_dice',
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

def get_output_filenames(out_path, args):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}{}_OUT{}".format(out_path, pathsplit[0][-21:], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask).astype(np.uint8))

def save_mask_and_gt(ori_img, mask_gt, mask, output_path, num_cls):
    '''

    :param mask_gt: origin mask ground truth
    :param mask: origin sized predicted mask
    :param output_path: the path which is to be saved
    '''
    # final_mask = label_map[mask]
    # true_mask = label_map[mask_gt]
    gt_image = ori_img.copy()
    pred_image = ori_img.copy()
    mask_gt = mask_gt.astype(np.uint8)
    mask = mask.astype(np.uint8)

    ori_h, ori_w = mask_gt.shape[:2]
    final_output = np.zeros((ori_h, 2 * ori_w + 10, 3), dtype=np.uint8)
    color = [(), (84, 255, 159), (171, 0, 96)]
    for i in range(1, num_cls):
        mask_gti = (mask_gt == i).astype(np.uint8)
        _, thresh = cv2.threshold(mask_gti, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, 3, 2)
        cv2.drawContours(gt_image, contours, -1, color[i], 2)
        # cnt = contours[0]
        # approx = cv2.approxPolyDP(cnt, 3, True)
        # cv2.polylines(gt_image, [approx], True, (84, 255, 159), 2)

        maski = (mask == i).astype(np.uint8)
        _, thresh = cv2.threshold(maski, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, 3, 2)
        cv2.drawContours(pred_image, contours, -1, color[i], 2)

    final_output[:, 0:ori_w, :] = gt_image
    # cnt = contours[0]
    # approx = cv2.approxPolyDP(cnt, 3, True)
    # cv2.polylines(pred_image, [approx], True, (84, 255, 159), 2)
    final_output[:, ori_w + 10:, :] = pred_image
    final_output[:, ori_w:ori_w + 10, :] = np.array([255, 255, 255])

    result = mask_to_image(final_output)
    result.save(output_path)
    print("Mask example saved to {}".format(output_path))

def generate_mask(mask, output_path):
    result = mask_to_image(mask)
    result.save(output_path)
    print('Mask saved to {}'.format(output_path))

def save_phase1_mask(mask_gt, mask, ori_img, output_img, output_mask, offset=30):
    '''

    :param mask_gt: origin mask ground truth
    :param mask: origin sized predicted mask
    :param output_path: the path which is to be saved
    '''
    global warning_num
    ori_h, ori_w, _ = ori_img.shape
    xmin = max(0, np.where(mask == 1)[0].min() - offset)
    ymin = max(0, np.where(mask == 1)[1].min() - offset)
    xmax = min(ori_w - 1, np.where(mask == 1)[0].max() + offset)
    ymax = min(ori_h - 1, np.where(mask == 1)[1].max() + offset)
    bbox = [xmin, ymin, xmax, ymax]
    # print('Bbox of the minimum area of model prediction: {}'.format(bbox))

    if mask_mapping[mask_gt].max() <= 0:
        final_mask_gt = mask_to_image(mask_gt)
        final_img = mask_to_image(ori_img)
    else:
        ori_xmin = np.where(mask_mapping[mask_gt] == 1)[0].min()
        ori_ymin = np.where(mask_mapping[mask_gt] == 1)[1].min()
        ori_xmax = np.where(mask_mapping[mask_gt] == 1)[0].max()
        ori_ymax = np.where(mask_mapping[mask_gt] == 1)[1].max()
        ori_bbox = [ori_xmin, ori_ymin, ori_xmax, ori_ymax]

        if ori_xmin < xmin or ori_ymin < ymin or ori_xmax > xmax or ori_ymax > ymax:
            print('This prediction has some problems: {}'.format(output_img))
            print('Ori bbox: {}'.format(ori_bbox))
            print('Predicted bbox: {}'.format(bbox))
            warning_num += 1

        final_mask_gt = mask_to_image(mask_gt).crop(bbox)
        final_img = mask_to_image(ori_img).crop(bbox)

    final_mask_gt.save(output_mask)
    final_img.save(output_img)

    print("Mask saved to {}".format(output_mask))
    print("Img saved to {}".format(output_img))

if __name__ == "__main__":
    args = get_args()

    generate_tag = False
    phase_name = None
    if args.phase:
        phase_name = 'phase1_plus'

    # data_set_root = '/home/mxj/data/Cervix/Segmentation/first_batch_phase2_resize_600'
    data_set_root = '/data/mxj/Cervix/Segmentation/cervix_resize_600_segmentation'
    net = unet34(num_classes=args.num_classes, criterion=args.loss, activation=args.activation)

    model_path = './checkpoints/cervix_resize_600_segmentation/{}/' \
                 'cross_entropy_1_10_10_transfer_pos_cls3_1_10_10_mar14new'.format(args.data_type)
    model_name = '{}_checkpoint.pth'.format(args.model)
    model_file = os.path.join(model_path, model_name)
    output_path = os.path.join(model_path, '{}_demo'.format(args.test_set))

    assert args.save_phase * args.save_mask == False
    if args.save_phase:
        img_phase1_path = os.path.join(data_set_root, args.data_set_name, '_'.join(['Images', 'phase1_plus']))
        mask_phase1_path = os.path.join(data_set_root, args.data_set_name, '_'.join(['Masks', 'phase1_plus']))
    if args.save_mask:
        data_set_root = '/home/mxj/data/Cervix/Classification/Lesion_Ternary'
        args.data_set_name = 'cervixMore_resize_for_fubao'
        output_path = os.path.join(data_set_root, args.data_set_name, 'Masks')
        generate_tag =True
    if args.draw:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
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

    test_dataset = Cervix(root=data_set_root, data_set_type=args.test_set, transform=get_valid_normalization,
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
    dice_lists = []

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
        if args.batch_size == 1:
            meta_info = meta_info[0]
        mask, result = predict_img(net=net,
                           img=img,
                           mask=true_mask,
                           test_strategy=test_strategy,
                           meta_info=meta_info)
        dice_list = result['dice']
        dice_lists.append(dice_list)
        acc = result['acc']
        recall_list = result['recall']
        # test_predict_per_img_list.extend([pred.astype(np.uint8) for pred in ((mask.reshape(-1) > 0).sum() > test_strategy['neg_thresh'])])
        test_predict_per_img_list.extend([(mask.reshape(-1) > 0).sum() > test_strategy['neg_thresh']])

        if args.draw:
            ori_img_name = meta_info['img_path'].split('/')[-1].strip('.jpg')
            if args.save_phase:
                out_img = os.path.join(img_phase1_path, ori_img_name+'.jpg')
                out_mask = os.path.join(mask_phase1_path, ori_img_name+'.gif')
                # if os.path.exists(out_img) and os.path.exists(out_mask):
                #     continue
            elif args.save_mask:
                out_mask = os.path.join(output_path, ori_img_name+'.gif')
                # if os.path.exists(out_mask):
                #     continue
            else:
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
                # if os.path.exists(out_file):
                #     continue

        if not args.save_mask:
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

        if args.draw:
            mask_gt = Image.open(meta_info['mask_path'])
            # if meta_info[0]['bbox'] is not None:
            #     mask_gt = mask_gt.crop(meta_info[0]['bbox'])
            mask_gt = np.array(mask_gt, dtype=np.int64)

            if args.save_phase:
                ori_img = np.array(Image.open(meta_info['img_path']), dtype=np.uint8)
                save_phase1_mask(mask_gt, mask, ori_img, out_img, out_mask)
            elif args.save_mask:
                generate_mask(mask, out_mask)
            else:
                if args.mask_type == 'Binary':
                    mask_gt = mask_mapping[mask_gt]
                elif args.mask_type == 'Real':
                    pass
                else:
                    raise ValueError
                ori_img = Image.open(meta_info['img_path'])
                ori_img = np.array(ori_img, dtype=np.uint8)
                save_mask_and_gt(ori_img, mask_gt, mask, out_file, args.num_classes)

    if not args.save_mask:
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
    else:
        print('Generate over!')
    print(np.array(test_target_per_img_list).sum(), len(test_target_per_img_list))


