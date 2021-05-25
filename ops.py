import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self, operators):
        super(MyBlock, self).__init__()
        self.operators = operators

    def forward(self, x):
        return self.operators(x)

class ConvBn(nn.Module):
	def __init__(self, inp: int, oup: int, stride: int):
		super(ConvBn, self).__init__()

		self.nnstack = nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

	def forward(self, x):
		x = self.nnstack(x)
		return x

class ConvDw(nn.Module):
	def __init__(self, inp: int, oup: int, stride: int):
		super(ConvDw, self).__init__()
		self.nnstack = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            ) 

	def forward(self, x):
		return self.nnstack(x)

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, stride=2):
        super(Unpool, self).__init__()

        self.stride = stride
        # create kernel [1, 0; 0, 0]
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:,:,0,0] = 1

    def forward(self, x):
        assert x.dim() == 4
        num_channels = x.size(1)
        return nn.ConvTranspose2d(x,
            self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1), kernel_size=5,
            stride=self.stride, groups=num_channels)

class UpConv(nn.Module):
	def __init__(self, inp: int, oup: int):
		super(UpConv, self).__init__()
		self.nnstack = nn.Sequential(
        Unpool(2),
        nn.Conv2d(inp,oup,kernel_size=5,stride=1,padding=2,bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(),
    )

	def forward(self, x):
		return self.nnstack(x)


class PointWise(nn.Module):
	def __init__(self, inp: int, oup: int):
		super(PointWise, self).__init__()
		self.nnstack = nn.Sequential(
          nn.Conv2d(inp,oup,1,1,0,bias=False),
          nn.BatchNorm2d(oup),
          nn.ReLU(inplace=True),
        )

	def forward(self, x):
		return self.nnstack(x)


class UpProj(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels):
        super(UpProj, self).__init__()
        #self.unpool = Unpool(2)
        #self.unpool = nn.MaxUnpool2d(2, )
        self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return self.relu(x1 + x2)


class ShuffleConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(ShuffleConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 256))
            self.conv2 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 64))
            self.conv3 = nn.Sequential(
                depthwise(16, kernel_size),
                pointwise(16, 16))
            self.conv4 = nn.Sequential(
                depthwise(4, kernel_size),
                pointwise(4, 4))
        else:
            self.conv1 = conv(256, 256, kernel_size)
            self.conv2 = conv(64, 64, kernel_size)
            self.conv3 = conv(16, 16, kernel_size)
            self.conv4 = conv(4, 4, kernel_size)

    def forward(self, x):
        x = F.pixel_shuffle(x, 2)
        x = self.conv1(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv2(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv3(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv4(x)

        x = F.pixel_shuffle(x, 2)
        return x

