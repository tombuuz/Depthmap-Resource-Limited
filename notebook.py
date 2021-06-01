import random
import json
import math
import time

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import random
import numpy as np

from mymodel import MobileNet
from mynyudata import NyuDataset
import myutils
from ops import ConvBn, ConvDw, UpConv, PointWise, UpProj, MyBlock, DeConvDw, NNConvDw, ShuffleConvDw
import torch.nn as nn

# Model
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convbn = ConvBn(3, 32, 2)
        self.convdw1 = ConvDw( 32,  64, 1)
        self.convdw2 = ConvDw( 64, 128, 2)
        self.convdw3 = ConvDw(128, 128, 1)
        self.convdw4 = ConvDw(128, 256, 2)
        self.convdw5 = ConvDw(256, 256, 1)
        self.convdw6 = ConvDw(256, 512, 2)
        self.convdw7 =  ConvDw(512, 512, 1)
        self.convdw8 =  ConvDw(512, 512, 1)
        self.convdw9 = ConvDw(512, 512, 1)
        self.convdw10 =  ConvDw(512, 512, 1)
        self.convdw11 = ConvDw(512, 512, 1)
        self.convdw12 = ConvDw(512, 1024, 2)
        self.convdw13 = ConvDw(1024, 1024, 1)    
        
        self.upconv1 = UpProj(1024, 512)
        self.upconv2 = UpProj(512, 256)
        self.upconv3 = UpProj(256, 128)
        self.upconv4 = UpProj(128, 64)
        self.upconv5 = UpProj(64, 32)
        self.convf = PointWise(32, 1)

    def forward(self, x):
        x = self.convbn(x)        
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

        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.convf(x)
        return x



class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean( torch.abs(grad_real-grad_fake) )

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x
def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)


def getnames(location):
    files = []
    for folder in os.listdir(location):
        if os.path.isdir('{}/{}'.format(location, folder)):
            for item in os.listdir('{}/{}'.format(location, folder)):
                if item.endswith('.h5'):
                    files.append('{}/{}/{}'.format(location, folder, item))
    return files



if __name__ == '__main__':
	# Locations
	LOAD_DIR = '.'

	# Hyperparameters
	bs = 8
	lr = 0.001

	# Settings
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    location = "/home/dlkhagvazhav/data/nyudepthv2/train"
    filenames = getnames(location)
    random.shuffle(filenames)
    train_size = 5000
    test_size = 1000
    trainset = NyuDataset(filenames[:train_size], train=True)
	valset = NyuDataset(filenames[train_size:train_size+test_size], train=False)

	trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)
	valloader = DataLoader(valset, batch_size=bs, shuffle=True)

	model = MyNet().to(DEVICE)

	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
	grad_criterion = GradLoss()


	# Training
	n_epoch = 50
	for epoch in range(n_epoch):
	    try:
	        torch.cuda.empty_cache()
	    except:
	        pass
	    model.train()
	    start = time.time()
	    # learning rate decay
	    if epoch > 5:
	        for param_group in optimizer.param_groups:
	            param_group['lr'] = param_group['lr'] * 0.9
	            
	    for i, (data, z) in enumerate(trainloader):
	        data, z = Variable(data.to(DEVICE)), Variable(z.to(DEVICE))
	        optimizer.zero_grad()
	        z_fake = model(data)
	        grad_real, grad_fake = imgrad_yx(z), imgrad_yx(z_fake)
	        loss = grad_criterion(grad_fake, grad_real)
	        loss.backward()
	        optimizer.step()
	        if (i+1) % 50 == 0:
	            print("[epoch %2d][iter %4d] loss: %.4f" % (epoch, i, loss))
	        
	    #Save model
	    torch.save(model.state_dict(),'{}/fyn_model.pt'.format(LOAD_DIR))
	    end = time.time()
	    print('model saved')
	    print('time elapsed: %fs' % (end - start))
	    
	    #Evaluation
	    if (epoch+1) % 5 == 0:
	        try:
	            torch.cuda.empty_cache()
	        except:
	            pass
	        model.eval()
	        print('evaluating...')
	        eval_loss = 0
	        count = 0
	        with torch.no_grad():
	            for i,(data,z) in enumerate(valloader):
	                data, z = Variable(data.to(DEVICE)), Variable(z.to(DEVICE))
	                z_fake = model(data)
	                eval_loss += float(data.size(0)) * grad_criterion(z_fake, z).item()**2
	                count += float(data.size(0))
	        print("[epoch %2d] RMSE_log: %.4f" % (epoch, math.sqrt(eval_loss/count)))
