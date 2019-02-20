# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pathlib
import numpy as np
import imageio
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class_size = 956

def read_txt_to_label(fname):
    label = []
    with open(fname,'r') as f:
        for line in f:
            label.append(int(line))
    return label

def transform_x(x):
    # normalize pixel values of the image to be in [0,1] instead of [0,255]
    xp = (x * (1. / 255)-0.5)*2.
    return xp


def transform_y(y,y_list):
    yp = np.zeros((y.shape[0],len(y_list)))
    for idx,label in enumerate(y_list):
        yp[:,idx] = (y==label).flatten()
    return yp


def get_ss_indices_per_class(y, sup_per_class):
    # number of indices to consider
#    n_idxs = y.size()[0]

    # calculate the indices per class
#    idxs_per_class = {j: [] for j in range(class_size)}

    # for each index identify the class and add the index to the right class
#    for i in range(n_idxs):
#        curr_y = y[i]
#        for j in range(class_size):
#            if curr_y[j] == 1:
#                idxs_per_class[j].append(i)
#                break
#
#    idxs_sup = []
#    idxs_unsup = []
#    for j in range(class_size):
#        np.random.shuffle(idxs_per_class[j])
#        idxs_sup.extend(idxs_per_class[j][:sup_per_class])
#        idxs_unsup.extend(idxs_per_class[j][sup_per_class:len(idxs_per_class[j])])
    idxs_sup = []
    idxs_unsup = []
    for j in range(class_size):
        idxs_per_class = np.nonzero(y[:,j])[0]
        np.random.shuffle(idxs_per_class)
        idxs_sup.extend(idxs_per_class[:sup_per_class])
        idxs_unsup.extend(idxs_per_class[sup_per_class:])
        

    return idxs_sup, idxs_unsup

def split_sup_unsup(X,y,sup_num):
    assert sup_num % class_size == 0, "unable to have equal number of images per class"

    # number of supervised examples per class
    sup_per_class = int(sup_num / class_size)

    idxs_sup, idxs_unsup = get_ss_indices_per_class(y, sup_per_class)
    X_sup = X[idxs_sup]
    y_sup = y[idxs_sup]
    X_unsup = X[idxs_unsup]
    y_unsup = y[idxs_unsup]

    return X_sup, y_sup, X_unsup, y_unsup

class ETL8(object):
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    
    def __init__(self,root,mode,sup_num=None,transform = None):
        self.root = root
        self.mode = mode
        assert mode in ["sup","unsup","valid","test","BASE"]
        
        ## read label
        label_paths = pathlib.Path(root).glob('*.txt')
        label_sorted = sorted([x for x in label_paths])
        label = read_txt_to_label(label_sorted[0])
        self.label_list = label
        
        ## transform definition
        def data_transform(x):
            return transform_x(x)
        def target_transform(y):
            return transform_y(y,self.label_list)
        
        print("Loading data\n")
        if mode in ["sup","unsup"]:
            ## read data
            data_paths = pathlib.Path(root).glob("train" + '/*.png')
            data_sorted = sorted([x for x in data_paths])
            data =[]
            labels = []
            for im_path in data_sorted:
                data.append(imageio.imread(str(im_path)))
                labels.append(label)
            
            data = np.concatenate(data,axis=1).reshape((64,-1,64,1)).transpose((1,3,0,2))
            labels = np.array(labels).reshape(-1,1)
            if data_transform is not None:
                data = data_transform(data)
            if target_transform is not None:
                labels = target_transform(labels)
            
        elif mode in ["valid","test"]:
            ## read label
            label_paths = pathlib.Path(root).glob('*.txt')
            label_sorted = sorted([x for x in label_paths])
            label = read_txt_to_label(label_sorted[0])
            ## read data
            data_paths = pathlib.Path(root).glob(mode + '/*.png')
            data_sorted = sorted([x for x in data_paths])
            data =[]
            labels = []
            for im_path in data_sorted:
                data.append(imageio.imread(str(im_path)))
                labels.append(label)
            
            data = np.concatenate(data,axis=1).reshape((64,-1,64,1)).transpose((1,3,0,2))
            labels = np.array(labels).reshape(-1,1)
            if data_transform is not None:
                data = data_transform(data)
            if target_transform is not None:
                labels = target_transform(labels)
        else:
            print("Lack of {} dataset".format(mode))
        
        
        ## split to sup and unsup
        print("Spliting data\n")
        if ETL8.train_data_sup is None:
            if sup_num is None:
                assert mode == "sup"
                ETL8.train_data_sup, ETL8.train_labels_sup = data, labels
            else:
                ETL8.train_data_sup, ETL8.train_labels_sup, \
                        ETL8.train_data_unsup, ETL8.train_labels_unsup =\
                        split_sup_unsup(data, labels, sup_num)
        
        if mode == "sup":
            self.data = ETL8.train_data_sup
            self.label = ETL8.train_labels_sup
        elif mode == "unsup":
            self.data = ETL8.train_data_unsup
            self.label = (torch.Tensor(
                    ETL8.train_labels_unsup.shape[0]).view(-1, 1)) * np.nan
        else:
            self.data = data
            self.label = labels
        
        self.transform = transform
        self.nitems = self.data.shape[0]
        
    def __getitem__(self,index):
        if self.data is None:
            self.data = None
        item = self.data[index]
        item_new = item
        if self.transform is not None:
            item_new = self.transform(item.transpose(1,2,0)).float()
        return item_new,torch.Tensor(self.label[index]).float()
    def __len__(self):
        return self.nitems

def setup_data_loaders(dataset,batch_size,sup_num=None,root=None,modes=["sup","unsup","valid","test"]):
    if root is None:
        root = '/Users/TrungKien/Documents/Workspace/Cinnamon/Data/etl_reg_numchar_956_size_64/'
    loaders = {}
    ETL_data = {}
    
    for mode in modes:
        ETL_data[mode] = dataset(root=root, mode=mode,sup_num=sup_num,transform = transforms.ToTensor())
        loaders[mode] = DataLoader(ETL_data[mode], batch_size=batch_size, shuffle=True)
    return loaders
    

if __name__ == "__main__":
    ## ARGS
    batch_size = 1000
    
    data_loaders = setup_data_loaders(ETL8,batch_size = batch_size,sup_num=956*50)
    ## test to read data
    mode = "sup"
    idata = iter(data_loaders[mode])
    #import matplotlib.pyplot as plt 
    X,Y = idata.next()
    save_image(X[:160].cpu(),mode + ".png", nrow=16)
    
