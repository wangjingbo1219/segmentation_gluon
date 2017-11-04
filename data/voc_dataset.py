import mxnet as mx
import mxnet.gluon as gluon
import os
from PIL import Image
import cfg
import random
import numpy as np
import augment


class VOCDataset(gluon.data.Dataset):
    def __init__(self, root, split, transform):
        super(VOCDataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self._img_path = os.path.join(root, 'img', '{}.jpg')
        self._label_path = os.path.join(root, 'SegmentationClass', '{}.png')
        self.ids = list()

        for line in open(os.path.join(root, split + '_seg.txt')):
            self.ids.append(line.strip())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = self._img_path.format(self.ids[idx])
        img = Image.open(img_path).convert('RGB')
        lbl_path = self._label_path.format(self.ids[idx])
        lbl = Image.open(lbl_path)

        img, lbl = self.transform(img, lbl)

        return img, lbl


def voc_test():
    dataset = VOCDataset(cfg.voc_root, 'train', augment.voc_train)
    print(len(dataset))
    dataloader = gluon.data.DataLoader(dataset, 4, True)

    for data in dataloader:
        print(data)
        break


if __name__ == '__main__':
    voc_test()
