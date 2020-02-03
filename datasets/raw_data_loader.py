from __future__ import print_function, absolute_import
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path, with_mask=False, mask_ext='png'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((128, 256), Image.ANTIALIAS)
            # calculate mask
            got_img = True
        except IOError as err:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

    return img, img, None


class RawImageData(Dataset):
    def __init__(self, dataset, transform, if_train=False, with_mask=False):
        self.dataset = dataset
        self.transform = transform
        self.with_mask = with_mask
        self.if_train = if_train
        self.path=[]
        for i in dataset:
            self.path.append(os.path.basename(i[0]))

    def __getbypath__(self,p):
        item=self.path.index(p)
        return self.__getitem__(item)

    def __getitem__(self, item):
        img_p, pid, camid, _ = self.dataset[item]
        img, fg_img, _ = read_image(img_p, self.with_mask)
        
        if self.if_train:
            if self.transform is not None:
                img = self.transform(img)
                fg_img = self.transform(fg_img)
            return img, pid, camid, fg_img, None,img_p
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, pid, camid,img_p

    def __len__(self):
        return len(self.dataset)

