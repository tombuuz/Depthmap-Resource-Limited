import torch
import torch.nn as nn
#import nni.retiarii.nn.pytorch as nn
#from nni.retiarii import basic_unit
from ops import ConvBn, ConvDw, UpConv, PointWise, UpProj, MyBlock
#from nni.nas.pytorch import mutables
from nni.nas import pytorch as nas 
#from nni.retiarii.nn.pytorch import LayerChoice
#import collections

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convbn = ConvBn(3, 32, 2)
        #self.convbn2 = ConvBn(3, 32, 4)

        #operators = nas.mutables.LayerChoice([ConvBn(3, 32, 2), ConvBn(3, 32, 2)], key='first_layer')
        #self.block = MyBlock(operators)
        
        #self.layer = LayerChoice(collections.OrderedDict([
        #    ("conv2b", ConvBn(3, 32, 2)), ("conv2b2", ConvBn(3, 32, 2))]))

        self.convdw1 = ConvDw( 32,  64, 1)
        self.convdw2 = ConvDw( 64, 128, 2)
        self.convdw3 = ConvDw(128, 128, 1)
        self.convdw4 = ConvDw(128, 256, 2)
        #self.convdw5 = ConvDw(256, 256, 1)
        operators = nas.mutables.LayerChoice([ConvDw(256, 256, 1), ConvDw(256, 256, 1)], key='first_layer')
        self.convdw5 = MyBlock(operators)

        self.convdw6 = ConvDw(256, 512, 2)
        self.convdw7 =  ConvDw(512, 512, 1)
        self.convdw8 =  ConvDw(512, 512, 1)
        self.convdw9 = ConvDw(512, 512, 1)
        self.convdw10 =  ConvDw(512, 512, 1)
        self.convdw11 = ConvDw(512, 512, 1)
        self.convdw12 = ConvDw(512, 1024, 2)
        self.convdw13 = ConvDw(1024, 1024, 1)
        
        """
        self.model = nn.Sequential(
            ConvBn(  3,  32, 2), 
            ConvDw( 32,  64, 1),
            ConvDw( 64, 128, 2),
            ConvDw(128, 128, 1),
            ConvDw(128, 256, 2),
            ConvDw(256, 256, 1),
            ConvDw(256, 512, 2),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 512, 1),
            ConvDw(512, 1024, 2),
            ConvDw(1024, 1024, 1),
            #nn.AvgPool2d(7),
        )
        """
        #self.fc = nn.Linear(1024, 1000)

        
        self.upconv1 = UpProj(1024, 512)
        self.upconv2 = UpProj(512, 256)
        self.upconv3 = UpProj(256, 128)
        self.upconv4 = UpProj(128, 64)
        self.upconv5 = UpProj(64, 32)
        self.convf = PointWise(32, 1)
        

    def forward(self, x):
        x = self.convbn(x)
        #x = self.layer(x)
        
        x = self.convdw1(x)
        x = self.convdw2(x)
        x = self.convdw3(x)
        x = self.convdw4(x)
        x = self.convdw5(x)
        x = self.convdw6(x)
        x = self.convdw7(x)
        x = self.convdw8(x)
        x = self.convdw9(x)
        x = self.convdw10(x)
        x = self.convdw11(x)
        x = self.convdw12(x)
        x = self.convdw13(x)

        #x =  self.fc1(x)
        # x = self.model(x)
        #x = x.view(-1, 1024)
        #x = self.fc(x)

        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.convf(x)
        return x

