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

def printr(string):
    sys.stdout.write("\r\033")
    sys.stdout.write(string)
    sys.stdout.flush()

def text_list_reader(flist, delimiter="|"):
    ## output list initialization
    imlist = []
    ## iteration on list of text file
    for f in flist:
        with open(f, 'r', encoding='utf-8') as rf:
            for line in rf.readlines():
                index = line.strip().index(delimiter)
                impath = line[0:index].strip()
                imlabel = line[index+1:].strip()
                ## add to output
                imlist.append((impath, imlabel))

    return imlist
    
def json_list_reader(flist):
    ## output list initialization
    imlist = []
    ## iteration on list of text file
    for f in flist:
        with open(f, 'r', encoding='utf-8') as rf:
            label_dict = json.load(rf)
            for impath, imlabel in label_dict.items():
                imlist.append((impath, imlabel))
            
    return imlist
    
def image_loader(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    #image = Image.open(path)
    return image#Image.open(path)
    
def label_loader(label):
    return label
    

class Loader(Dataset):
    def __init__(self, flist, 
                        root=None, 
                        transform=None, target_transform=None, common_transform=None,
                        flist_reader=text_list_reader, 
                        image_loader=image_loader,
                        label_to_image=False,
                        delimiter=None):
        self.root = root
        ## imlist loader
        if delimiter is not None:
            self.imlist = flist_reader(flist, delimiter=delimiter)
        else: 
            self.imlist = flist_reader(flist)
        ## transform
        self.image_loader = image_loader
        self.label_to_image = label_to_image
        self.transform = transform
        self.target_transform = target_transform
        self.common_transform = common_transform
        
    def __len__(self):
        return len(self.imlist)
    
    def _label_to_image(self, label_dict, imsize):
        im_out = np.zeros(imsize[:2]).astype(np.uint8)
        for imline, imloc in label_dict.items():
            cv2.drawContours(im_out, [np.array(imloc)], 0,
                                        255, -1)
        return Image.fromarray(im_out)
    
    def _common_transform(self, image, target):
        seed = random.randint(0, 1000)
        random.seed(seed)
        img = self.common_transform(image)
        random.seed(seed)
        target = self.common_transform(target)

        return img, target

    
    def __getitem__(self, index):
        impath, target = self.imlist[index]
        #print("Image: ", impath, "--> label: ", target)
        img = self.image_loader(os.path.join(self.root, impath))
        if self.label_to_image:
            target = self._label_to_image(target, np.array(img).shape)
        
        if self.common_transform is not None:
            img, target = self._common_transform(img, target)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
        
    

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