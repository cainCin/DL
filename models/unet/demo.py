import sys
sys.path.insert(0, "../../lib_cain")
from loader import Loader, json_list_reader
from trainer import Trainer
from unet import UNet
import torch
import numpy as np
import cv2
import os
from pathlib import Path

from torchvision.utils import save_image
import torch.nn.functional as F
"""
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

from albumentations.augmentations.transforms import (Resize, HorizontalFlip, VerticalFlip, RandomCrop)
# transform
def strong_aug(p=0.5):
    return Compose([
        Resize(140,140),
        OneOf([
            HorizontalFlip(),
            RandomRotate90(),
            VerticalFlip(),
        ], p=0.2),
        RandomCrop(128,128),
    ], p=p)
augmentation = strong_aug(p=0.9)
"""
from torchvision import transforms
from transform import *


# utility
def imread_unicode(impath, mode=cv2.IMREAD_COLOR):
    stream = open(impath, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, mode)
    return img
# refactor
class UnetTrainer(Trainer):
    def _get_acc(self, xs, ys):
        ## feed into device
        xs = xs.to(self.device)
        ys = ys.to(self.device)
        # get loss
        pred = self.model["unet"](xs)
        loss = self.criterion(pred, ys)*xs.shape[0]
        return loss
    
    def _debug(self, loader, debug_dir="debug/val"):
        #debug_dir = "debug"
        if not os.path.isdir(debug_dir):
            os.makedirs(debug_dir)
        with torch.no_grad():
            for i, (data, label) in enumerate(loader):
                # feed to device
                data = data.to(self.device)
                pred = self.model["unet"](data)
                # save to debug
                out = os.path.join(debug_dir, "%d.jpg" %i)
                #out_img = (np.squeeze(pred.cpu().numpy())*255).astype(np.uint8)
                save_image(torch.cat([pred.cpu(), label]), out, nrow=2)
                #cv2.imwrite(out, out_img)
    
    def test(self, test_path, test_dir="debug/test"):
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        EXTENSIONS = ['.png', '.jpeg', '.jpg']
        for data in Path(test_path).glob('**/*'):
            if data.suffix.lower() not in EXTENSIONS:
                continue
            img_name = os.path.basename(str(data))
            img = imread_unicode(str(data), cv2.IMREAD_COLOR)
            ## image resize
            #image = cv2.resize(img, (128,128))
            image = Image.fromarray(img)
            input = test_trans(image).view(1,3,128,128)#transforms.ToTensor()(image).view(1,3,128,128)
            input = input.to(self.device)
            out_img = self.model["unet"](input)
            out = os.path.join(test_dir, img_name)
            
            save_image(torch.cat([input[:,:1], out_img]), out)
            
    
                
    def export(self, name=None):
        for key, mdl in self.model.items():
            if name is None:
                torch.save(mdl, key)
            else:
                torch.save(mdl, name + key)

# transform definition
common_trans = transforms.Compose([
        transforms.Resize((140, 140)),
        transforms.RandomChoice([
            transforms.RandomRotation(45, expand=True),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5), 
        ]),
        transforms.RandomCrop((128, 128)),
        ])


trans = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.5),
        cain_Gradient(),
        transforms.ToTensor(),
        cain_Normalize(),
        ])
        
test_trans = transforms.Compose([
        transforms.Resize((128, 128)),
        cain_Gradient(),
        transforms.ToTensor(),
        cain_Normalize(),
        ])
target_trans = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        ])
# parameters
train_flist = ["D:/Workspace/Cinnamon/Data/Deloitte/data_unet/train/labels_refactor.json"]
train_root = "D:/Workspace/Cinnamon/Data/Deloitte/data_unet/train/images"
test_flist = ["D:/Workspace/Cinnamon/Data/Deloitte/data_unet/test/labels_refactor.json"]
test_root = "D:/Workspace/Cinnamon/Data/Deloitte/data_unet/test/images"
# load data
train_dataset = Loader(train_flist, root=train_root, 
                    flist_reader=json_list_reader,
                    label_to_image=True,
                    common_transform=common_trans,
                    transform=trans,
                    target_transform=target_trans,
                    )
test_dataset = Loader(test_flist, root=test_root, 
                    flist_reader=json_list_reader,
                    label_to_image=True,
                    #common_transform=common_trans,
                    transform=test_trans,
                    target_transform=target_trans)

data_loaders = {"train": torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True),
                "val": torch.utils.data.DataLoader(test_dataset, batch_size=1)}
"""
for i, (data, label) in enumerate(data_loaders["val"]):
    #print(data.shape, label.shape)
    save_image(torch.cat([data[:,:1], label]), "test%02d.png" %i, nrow=4)"""



               
# model
device = torch.device("cuda")
model = {"unet": UNet(3,1).to(device)}  
# criterion
criterion = torch.nn.BCELoss()
# optimizer
opt = {"adam": torch.optim.Adam(model["unet"].parameters(), lr=0.0001)}
# trainer
trainer = UnetTrainer(model=model, 
                    optimizer=opt, 
                    criterion=criterion,
                    device=device,
                    checkpoint="unet_128_aug_bright05_contrast05_hue05_grad.pth")
#trainer._debug(data_loaders["val"]) # test if model is loaded
# checkpoint setup without loading in initilization
#trainer.checkpoint = "unet_128_aug_bright05_contrast05_hue05_grad.pth"
#trainer.export()

test_path = "D:/Workspace/Cinnamon/Data/Deloitte/raw data/Raw Testing data/02. Shinkansen"

for epoch in range(50): # run debug for each epoch
    trainer.train(dataloaders=data_loaders, epochs=1)
    # store checkpoint
    trainer._save_checkpoint()
    # debug
    print("disp train set")
    trainer._debug(data_loaders["train"], "debug/train")
    print("disp val set")
    trainer._debug(data_loaders["val"], "debug/val")
    # test
    print("disp test set")
    trainer.test(test_path, "debug/test")

