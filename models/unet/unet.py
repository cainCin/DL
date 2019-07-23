# full assembly of the sub-parts to form the complete net

import torch
import torch.nn.functional as F

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down2_2 = down(128, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3_2 = self.down2_2(x3)
        x = self.up2(x3_2, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

def test():
    import time
    input = torch.zeros([1, 3,480,480]).to("cpu")
    model = UNet(3,1).to("cpu")
    now = time.time()
    output = model(input)
    print(time.time()-now)

    now = time.time()
    print(time.time()-now)

    now = time.time()
    print(time.time()-now)

    now = time.time()
    output = model(input)
    print(time.time()-now)

    now = time.time()
    output = model(input)
    print(time.time()-now)

if __name__ == "__main__":
    test()