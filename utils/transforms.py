import numpy as np
import cv2
import math
import random
import torch

from albumentations import ElasticTransform


class Compose(object):
    """Compose transforms together."""

    def __init__(self, transforms, p=1.0):
        self.transforms = [t for t in transforms if t is not None]
        self.p = p

    def __call__(self, **data):
        if random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
        return data


class OneOf(object):
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, **data):
        if random.random() < self.p:
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            t.p = 1.
            data = t(**data)
        return data


class BasicTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **kwargs):
        if random.random() < self.p:
            params = self.get_params(**kwargs)
            params = self.update_params(params, **kwargs)
            return {key: self.targets.get(key, lambda x, **p: x)(arg, **params) for key, arg in kwargs.items()}
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self, **kwargs):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params


class DualTransform(BasicTransform):
    """Transform for segmentation task."""

    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply_to_mask}

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    @property
    def targets(self):
        return {'image': self.apply}


class RandomCrop(DualTransform):
    def __init__(self, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.), p=1):
        super(RandomCrop, self).__init__(p)
        self.scale = scale
        self.ratio = ratio

    def get_params(self, **kwargs):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        img = kwargs['image']
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)
            np.random.randn()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return {'i_t': i,
                        'j_t': j,
                        'h_t': h,
                        'w_t': w}

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return {'i_t': i,
                'j_t': j,
                'h_t': w,
                'w_t': w}

    def apply(self, img, i_t=0, j_t=0, h_t=0, w_t=0, **params):
        cropped = img[i_t:i_t + h_t, j_t:j_t + w_t, :]
        return cropped


class ShiftScaleRotate(DualTransform):
    def __init__(self, scale=1, angle_max=15, dx=0, dy=0, p=0.5):
        super(ShiftScaleRotate, self).__init__(p)
        self.scale = scale
        self.angle_max = angle_max
        self.dx = dx
        self.dy = dy

    def get_params(self, **kwargs):
        angle = np.random.uniform(-self.angle_max, self.angle_max)
        return {'angle': angle}

    def apply(self, img, angle=None, **params):
        borderMode = cv2.BORDER_REFLECT_101
        # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

        height, width = img.shape[:2]
        sx = self.scale
        sy = self.scale

        cc = math.cos(angle / 180 * math.pi) * (sx)
        ss = math.sin(angle / 180 * math.pi) * (sy)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + self.dx, height / 2 + self.dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_NEAREST,
                                  borderMode=borderMode, borderValue=(
                0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        return img


class RandomShiftCrop(DualTransform):
    def __init__(self, limit=0.1, p=0.5):
        super(RandomShiftCrop, self).__init__(p)
        self.limit = limit

    def get_params(self, **kwargs):
        H, W = kwargs['image'].shape[:2]
        dy = int(H * self.limit)
        y0 = np.random.randint(0, dy)
        y1 = H - np.random.randint(0, dy)

        dx = int(W * self.limit)
        x0 = np.random.randint(0, dx)
        x1 = W - np.random.randint(0, dx)

        return {'x0': x0,
                'x1': x1,
                'y0': y0,
                'y1': y1}

    def apply(self, img, x0=0, x1=0, y0=0, y1=0, **params):
        return img[y0:y1, x0:x1]


class RandomBrightnessShift(ImageOnlyTransform):
    def __init__(self, limit=(-0.05, 0.05), p=0.5):
        super(RandomBrightnessShift, self).__init__(p)
        self.low = limit[0]
        self.high = limit[1]

    def get_params(self, **kwargs):
        shift = np.random.uniform(self.low, self.high)
        return {'shift': shift}

    def apply(self, img, shift=0, **params):
        if len(img.shape) == 2:
            return img
        img = img.astype(np.float32)
        img += shift
        return np.clip(img, 0, 255)


class RandomBrightnessMultiply(ImageOnlyTransform):
    def __init__(self, limit=(-0.05, 0.05), p=0.5):
        super(RandomBrightnessMultiply, self).__init__(p)
        self.low = limit[0]
        self.high = limit[1]

    def get_params(self, **kwargs):
        rate = np.random.uniform(1 + self.low, 1 + self.high)
        return {'rate': rate}

    def apply(self, img, rate=1, **params):
        if len(img.shape) == 2:
            return img
        img = img.astype(np.float32)
        img *= rate
        return np.clip(img, 0, 255)


class RandomBrightnessGamma(ImageOnlyTransform):
    def __init__(self, limit=(-0.05, 0.05), p=0.5):
        super(RandomBrightnessGamma, self).__init__(p)
        self.low = limit[0]
        self.high = limit[1]

    def get_params(self, **kwargs):
        gamma = np.random.uniform(1 + self.low, 1 + self.high)
        return {'gamma': gamma}

    def apply(self, img, gamma=1, **params):
        if len(img.shape) == 2:
            return img
        img = img.astype(np.float32)
        return np.clip(np.power(img, gamma), 0, 255)


class HorizontalFlip(DualTransform):
    def __init__(self, p=0.5):
        super(HorizontalFlip, self).__init__(p)

    def apply(self, img, **params):
        image = np.ascontiguousarray(img[:, ::-1])

        return image


class ReflectiveCenterPad(DualTransform):
    def __init__(self, up_rate, target_size, p=1):
        super(ReflectiveCenterPad, self).__init__(p)
        self.up_rate = up_rate
        self.target_size = target_size

    def apply(self, img, **params):
        ori_h, ori_w = img.shape[:2]
        # img_phase1 = cv2.resize(img, (self.up_rate * ori_w, self.up_rate * ori_h), interpolation=cv2.INTER_CUBIC)
        img_phase1 = cv2.resize(img, (self.up_rate * ori_w, self.up_rate * ori_h), interpolation=cv2.INTER_NEAREST)
        if len(img.shape) == 2:
            img_phase2 = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        elif len(img.shape) == 3:
            img_phase2 = np.zeros((self.target_size, self.target_size, 3), dtype=np.float32)
        else:
            raise ValueError
        sec_h, sec_w = img_phase1.shape[:2]
        left_boder = (self.target_size - sec_w) // 2
        top_boder = left_boder
        right_boder = self.target_size - sec_w - left_boder
        bottom_boder = self.target_size - sec_h - top_boder
        img_phase2[top_boder:top_boder + sec_h, left_boder:left_boder + sec_w] = img_phase1
        # left
        img_phase2[top_boder:top_boder + sec_h, :left_boder] = img_phase2[top_boder:top_boder + sec_h,
                                                               left_boder:2 * left_boder][:, ::-1]
        img_phase2[top_boder:top_boder + sec_h, left_boder + sec_w:] = img_phase2[top_boder:top_boder + sec_h,
                                                                       sec_w + left_boder - right_boder:sec_w + left_boder][
                                                                       :, ::-1]
        img_phase2[:top_boder, :] = img_phase2[top_boder:2 * top_boder, :][::-1, :]
        img_phase2[sec_h + top_boder:, :] = img_phase2[sec_h + top_boder - bottom_boder: sec_h + top_boder, :][::-1, :]

        return img_phase2


class Normalize(ImageOnlyTransform):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1):
        super(Normalize, self).__init__(p)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def apply(self, image, **params):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image


class Resize(DualTransform):
    def __init__(self, size, p=1):
        super(Resize, self).__init__(p)
        self.size = size

    def apply(self, image, **params):
        try:
            h, w, _ = image.shape
        except ValueError:
            h, w = image.shape
        # image = cv2.resize(image, (self.size, self.size))
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image


class ToTensor(ImageOnlyTransform):
    def __init__(self, p=1):
        super(ToTensor, self).__init__(p)

    def apply(self, img, **params):
        '''
        :param img: numpy.ndarray
        :return:
        '''
        return torch.from_numpy(img).float().permute(2, 0, 1)
