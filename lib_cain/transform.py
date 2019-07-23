import torch
import torch.nn as nn
import os, sys
from PIL import Image
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
import random
from torchvision.transforms import functional as F

## Numpy transform
def to_gradient(img):
    W, H, C = img.shape
    out = []
    for c in range(C):
        grad = np.abs(cv2.Laplacian(img[:,:,c], cv2.CV_64F))
        grad = (grad-grad.min()) / (grad.max()-grad.min())
        grad = (grad*255).astype(np.uint8)
        out.append(grad)
    out = np.stack(out, axis=-1)
    #print(out.shape)
    return out
    
class cain_Gradient(object):
    def __init__(self, to_grad=True):
        self.to_grad = to_grad
        
    
    def __call__(self, image):
        """
        Args:
            image (PIL): Tensor image of size (C, H, W) to be normalized.

        Returns:
            img: Normalized PIL image.
        """

        img = np.array(image, dtype=np.uint8)
        grad = to_gradient(img)
        grad = Image.fromarray(grad)
        return grad


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

## Tensor transform
class cain_Normalize(object):
    def __init__(self, mean=None, std=None, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        
    def _estimate(self, tensor):
        C, H, W = tensor.shape
        mean = tensor.view(C,-1).mean(dim=-1)
        std = tensor.view(C,-1).std(dim=-1)
        return tuple(mean.tolist()), tuple(std.tolist())
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        #if self.mean is None:
        self.mean, self.std = self._estimate(tensor)
        return F.normalize(tensor, self.mean, self.std, self.inplace)


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        



        
    

if __name__ == "__main__": ## test loader
    """
    root = ""
    flist = ["D:/Workspace/Cinnamon/Code/ref/rnd_fixed_form_hw_ocr-HW_FFG_field1_2/main_source/address-train1.0_resort.txt"]
    loader = Loader(flist, delimiter=',')
    for data in loader:
        print(data)"""

    
    ## barry converter
    
    flist = ["D:/Workspace/Cinnamon/Data/Deloitte/data_unet/test/labels.json"]
    ## iteration on list of text file
    out_dict = dict()
    for f in flist:
        new_f = f[:-5] + '_refactor' + f[-5:]
        with open(new_f, 'w', encoding='utf-8') as wf:
            with open(f, 'r', encoding='utf-8') as rf:
                label_dict = json.load(rf)
                for impath, imvalue in label_dict.items():
                    new_dict = dict()
                    for imline, imlocs in imvalue.items():
                        for i, imloc in enumerate(imlocs):
                            new_dict.update({'%s_%d' % (imline, i): imloc})
                            #print(impath, '[%s_%d]' % (imline, i), imloc)
                    #print(impath, new_dict)
                    out_dict.update({impath: new_dict})    
            json.dump(out_dict, wf, ensure_ascii=False, indent=2)
    """
    flist = ["D:/Workspace/Cinnamon/Data/Deloitte/data_unet/train/labels_refactor.json"]
    #json_list_reader(flist)
    root = "D:/Workspace/Cinnamon/Data/Deloitte/data_unet/train/images"
    loader = Loader(flist, root=root, 
                        flist_reader=json_list_reader)
    for i, data in enumerate(loader):
        print(data)
        if i > 10:
            break"""