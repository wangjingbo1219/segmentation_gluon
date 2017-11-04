from PIL import Image
import math
import random
import numpy as np
import mxnet as mx
import cfg


def resized_crop(img, lbl, i, j, h, w, size):
    img = img.crop((j, i, j + w, i + h))
    lbl = lbl.crop((j, i, j + w, i + h))
    img = img.resize((size, size), Image.BILINEAR)
    lbl = lbl.resize((size, size))

    return img, lbl


class RandomScale:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, lbl):
        scale = random.uniform(self.min_scale, self.max_scale)
        # target_area = scale * (img.size[0] * img.size[1])
        # ratio = img.size[0] / img.size[1]

        # w = int(round(math.sqrt(target_area * ratio)))
        # h = int(round(math.sqrt(target_area / ratio)))

        w = int(round(scale * img.size[0]))
        h = int(round(scale * img.size[1]))

        img = img.resize((w, h), Image.BILINEAR)
        lbl = lbl.resize((w, h))

        return img, lbl


class RandomResizedCrop:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img):
        # try to crop a random portion of the image with random ratio
        if random.random() < 0.8:
            for attempt in range(10):
                area = img.size[0] * img.size[1]
                target_area = random.uniform(0.4, 1.0) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if random.random() < 0.5:
                    w, h = h, w

                if w <= img.size[0] and h <= img.size[1]:
                    i = random.randint(0, img.size[1] - h)
                    j = random.randint(0, img.size[0] - w)
                    return i, j, h, w

        # Fallback to random crop or just resize the whole image

        if random.random() < 0.5:
            w = min(img.size[0], img.size[1])
            i = random.randint(0, img.size[1] - w)
            j = random.randint(0, img.size[0] - w)
            return i, j, w, w
        else:
            return 0, 0, img.size[1], img.size[0]

    def __call__(self, img, lbl):
        i, j, h, w = self.get_params(img)
        return resized_crop(img, lbl, i, j, h, w, self.size)


class RandomHorizontalFlip:
    def __call__(self, img, lbl):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl


class Resize:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img, lbl):
        return img.resize((self.w, self.h), Image.BILINEAR), lbl.resize((self.w, self.h))


class ToNDArray():
    def __call__(self, img, lbl):
        img = mx.nd.array(img)
        img = img / 255

        lbl = mx.nd.array(lbl)

        return img, lbl


class Normalize:
    def __init__(self, mean, std):
        self.mean = mx.nd.array(mean)
        self.std = mx.nd.array(std)

    def __call__(self, img, lbl):
        img = mx.image.color_normalize(img, self.mean, self.std)
        img = mx.nd.transpose(img, (2, 0, 1))

        return img, lbl


class Compose:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, lbl):
        for t in self.trans:
            img, lbl = t(img, lbl)
        return img, lbl


class RandomCrop:
    def __init__(self, crop_size=None, scale=None):
        # assert min_scale <= max_scale

        self.crop_size = crop_size
        self.scale = scale
        # self.min_scale = min_scale
        # self.max_scale = max_scale

    def __call__(self, img, lbl):
        if self.crop_size:
            crop = self.crop_size
        else:
            crop = min(img.size)
        if self.scale:
            factor = random.uniform(self.scale, 1.0)
            crop = int(round(crop * factor))

        x = random.randint(0, img.size[0] - crop)
        y = random.randint(0, img.size[1] - crop)

        img = img.crop((x, y, x + crop, y + crop))
        lbl = lbl.crop((x, y, x + crop, y + crop))
        return img, lbl


class UnitResize:
    def __init__(self, unit):
        self.unit = unit

    def __call__(self, img, lbl):
        w, h = img.size
        if w % self.unit == 0 and h % self.unit == 0:
            return img, lbl
        w = int(round(w / self.unit) * self.unit)
        h = int(round(h / self.unit) * self.unit)
        return img.resize((w, h), Image.BILINEAR), lbl.resize((w, h))


cityscapes_train = Compose([
    RandomCrop(crop_size=cfg.crop),
    Resize(cfg.size, cfg.size),
    RandomHorizontalFlip(),
    ToNDArray(),
    Normalize(cfg.mean, cfg.std),

])

cityscapes_val = Compose([
    Resize(960, 480),
    ToNDArray(),
    Normalize(cfg.mean, cfg.std),

])

cityscapes_test = Compose([
    UnitResize(32),
    ToNDArray(),
    Normalize(cfg.mean, cfg.std),

])

cityscapes_t = Compose([
    Resize(cfg.size * 2, cfg.size),
    ToNDArray(),
    Normalize(cfg.mean, cfg.std),
])
