import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
import utils as mutil
import kornia
from .rrdbnet import RRDB, RRDBNet


class EESNGenerator(nn.Module):
  def __init__(self, in_nc, out_nc, nf, nb):
    super(EESNGenerator, self).__init__()
    self.netRG = RRDBNet(in_nc, out_nc, nf, nb)
    self.netE = EESN()

  def forward(self, x):
    x_base, intermediate_in, intermediate_out = self.netRG(x) # add bicubic according to the implementation by author but not stated in the paper
    x5, x_lap = self.netE(x_base) # EESN net
    x_sr = x5 + x_base - x_lap

    return x_base, x_sr, x5, x_lap, intermediate_in, intermediate_out

'''
Create EESN from this paper: https://ieeexplore.ieee.org/document/8677274 EEGAN - Edge Enhanced GAN
'''
'''
Only EESN
'''
class EESN(nn.Module):
  def __init__(self):
    super(EESN, self).__init__()
    self.beginEdgeConv = BeginEdgeConv() #  Output 64*64*64 input 3*64*64
    self.denseNet = EESNRRDBNet(64, 256, 64, 5) # RRDB densenet with 64 in kernel, 256 out kernel and 64 intermediate kernel, output: 256*64*64
    self.maskConv = MaskConv() # Output 256*64*64
    self.finalConv = FinalConv() # Output 3*256*256

  def forward(self, x):
    x_lap = kornia.laplacian(x, 3) # see kornia laplacian kernel
    x1 = self.beginEdgeConv(x_lap)
    x2 = self.denseNet(x1)
    x3 = self.maskConv(x1)
    x4 = x3*x2 + x2
    x_learned_lap = self.finalConv(x4)

    return x_learned_lap, x_lap

'''
Starting layer before Dense-Mask Branch
'''
class BeginEdgeConv(nn.Module):
    def __init__(self):
        super(BeginEdgeConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        self.conv_layer4 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv_layer5 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)
        self.conv_layer6 = nn.Conv2d(256, 64, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3,
                            self.conv_layer4, self.conv_layer5, self.conv_layer6], 0.1)

    def forward(self, x):
      x = self.lrelu(self.conv_layer1(x))
      x = self.lrelu(self.conv_layer2(x))
      x = self.lrelu(self.conv_layer3(x))
      x = self.lrelu(self.conv_layer4(x))
      x = self.lrelu(self.conv_layer5(x))
      x = self.lrelu(self.conv_layer6(x))

      return x

'''
Dense sub branch
'''
class EESNRRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(EESNRRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        #fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.lrelu(self.conv_last(self.lrelu(self.HRconv(fea))))

        return out

'''
Second: Mask Branch of two Dense-Mask branch
'''
class MaskConv(nn.Module):
    def __init__(self):
        super(MaskConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3], 0.1)

    def forward(self, x):
      x = self.lrelu(self.conv_layer1(x))
      x = self.lrelu(self.conv_layer2(x))
      x = self.lrelu(self.conv_layer3(x))
      x = torch.sigmoid(x)

      return x

'''
Final conv layer on Edge Enhanced network
'''
class FinalConv(nn.Module):
    def __init__(self):
        super(FinalConv, self).__init__()

        self.upconv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.upconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.HRconv(x)))

        return x
