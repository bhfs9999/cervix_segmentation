# -*- coding: utf-8 -*-
import os
import os.path
import torch
import pickle

import numpy as np
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

type_mapping = {
    'acid'  : '_2',
    'iodine': '_3'
}

# all_mask_mapping = np.array([0, 1, 1, 1])
all_mask_mapping = np.array([0, 1, 1, 0])

class Cervix(Dataset):
    def __init__(self, root, data_set_type, transform, data_type=None, tf_learning=True, test_mode=False):
        self._root_path = root
        self._data_set_type = data_set_type
        self._data_type = data_type

        # self._post_transform = post_transform()

        self._test_mode = test_mode

        if tf_learning == True:
            self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        else:
            # need to calculate mean and std of the train_set
            # self._mean =
            # self._std =
            raise ValueError
        self._transform = transform(mean=self._mean, std=self._std)

        self._file_suffix = '.jpg'
        self._mask_suffix = '.gif'
        self._type_suffix = type_mapping[data_type]

        self._all_image_indexes = self._load_image_set_index()


    def __getitem__(self, index):
        return self._pull_item(index)

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return len(self._all_image_indexes)

    @property
    def data_set_type_file(self):
        return os.path.join(self._root_path, 'data_split', self._data_type, self._data_set_type + '.txt')

    @property
    def image_file_template(self):
        return os.path.join(self._root_path, 'Images', '{:s}' + self._file_suffix)

    @property
    def mask_file_template(self):
        return os.path.join(self._root_path, 'Masks', '{:s}' + self._mask_suffix)


    '''
    Data Load
    '''
    def _image_path_at(self, index_name):
        """
        Return the absolute path to image 'index_name' in the image sequence.
        """
        img_path = self.image_file_template.format(index_name)
        assert os.path.exists(img_path), 'Path does not exist: {}'.format(img_path)
        return img_path

    def _mask_path_at(self, index_name):
        mask_path = self.mask_file_template.format(index_name)
        assert os.path.exists(mask_path), 'Path does not exist: {}'.format(mask_path)
        return mask_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this data-set's image set file.
        """
        # Example path to image set file:
        image_set_file = self.data_set_type_file
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_image(self, index):
        img_path = self._image_path_at(index)
        # PIL Image
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img, img_path

    def _load_mask(self, index):
        mask_path = self._mask_path_at(index)
        # PIL Image
        mask = Image.open(mask_path)
        return mask, mask_path

    def _pull_item(self, index):
        ''' Load Image
			:param item:
			:return:
			'''
        item_index = self._all_image_indexes[index]
        img, img_path = self._load_image(item_index)
        mask, mask_path = self._load_mask(item_index)
        if self._test_mode:
            meta_info = {}
            # [H, W]
            meta_info['ori_size'] = [img.size[1], img.size[0]]
            meta_info['img_path'] = img_path
            meta_info['mask_path'] = mask_path

        # Data Augmentation / Base Transform, input needed to be numpy, return is numpy
        # replicate the image on the channel dim
        data = {'image': np.array(img, dtype=np.float32), 'mask': np.array(mask, dtype=np.float32)}
        pos_data = self._transform(**data)
        img = pos_data['image']
        mask = pos_data['mask']
        input_img = img

        mask = mask.astype(np.int64)
        # if binary mask needed, mask should transfer to binary form
        mask = all_mask_mapping[mask]

        target_mask = torch.LongTensor(mask)
        if not self._test_mode:
            return input_img, target_mask
        else:
            return input_img, target_mask, meta_info

    '''
    collate_fn of this data-set
    '''
    def classify_collate(self, batch):
        '''
        :param batch: [imgs, labels] dtype = np.ndarray
        imgs:
          shape = (C H W)
        :return:
        '''

        imgs = [x[0] for x in batch]
        masks = [x[1] for x in batch]
        if self._test_mode:
            meta_infos = [x[2] for x in batch]
            return torch.stack(imgs, 0), torch.stack(masks, 0), meta_infos
        return torch.stack(imgs, 0), torch.stack(masks, 0)