import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import os
from PIL import Image
import cfg
import random
import numpy as np


class ADEDataset(gluon.data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split =split

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
