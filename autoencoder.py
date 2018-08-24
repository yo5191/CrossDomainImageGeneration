
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.image as mpimg
import numpy as np
import imageio
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import time
from PIL import Image
import sys


TEST_DIR = "/home/student4/male_test/" #Folder contains test images - only males.
TRAIN_DIR = "/home/student4/male_train_new/" #Folder contains train images - only males.
SAVE_RESULTS_DIR = "/home/student4/link/ae8_results/" #Where you want to save your results
SAVE_AE_WEIGHTS = "/home/student4/link/ae8_results/new_autoencoder.pth" #saving generator weights

# Hyper Parameters
EPOCH = 450         #number of epochs
BATCH_SIZE = 64
LR = 0.0001        # learning rate
N_TRAIN_IMG = BATCH_SIZE * 1093 #Number of Train images
N_TEST_IMG = BATCH_SIZE * 1 #Number of Test images
HEIGHT = 128 #Dimensions of single image
WIDTH = 128
CHANNELS = 3
IMG_SIZE = HEIGHT * WIDTH * CHANNELS
IS_GPU = 1 #Please note the code only tested when IS_GPU=1
LOAD_WEIGHTS = 0 #Load the weights of the autoencoder
AE_WEIGHTS =  "/home/student4/Eliran/our_autoencoder.pth"

#DataSet class
class Faces(Data.Dataset):
    """Faces."""

    def __init__(self, root_dir, transform, size):
        self.root_dir = root_dir
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.size #number of images

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(idx)+".jpg")
        image = Image.open(img_name)
        sample = self.transform(image)
        return sample


transform = transforms.Compose(
    [transforms.Resize((HEIGHT, WIDTH)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


test_data = Faces(TEST_DIR, transform, N_TEST_IMG)
train_data = Faces(TRAIN_DIR, transform, N_TRAIN_IMG)
#Create the data loaders
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.ndf = 32
        latent_variable_size = 256
        self.e0 = nn.Conv2d(3, self.ndf, 3, 2, 1)
        self.bn0 = nn.BatchNorm2d(self.ndf)
        #self.encoder = nn.Sequential(
        self.e1 = nn.Conv2d(self.ndf, self.ndf, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(self.ndf)
        self.e2 = nn.Conv2d(self.ndf, self.ndf * 2, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.ndf * 2)
        self.e3 = nn.Conv2d(self.ndf * 2, self.ndf * 2, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.ndf * 2)

        #res block 1
        self.cr1 = nn.Conv2d(self.ndf * 2, self.ndf*2, 3, 2, 0)
        self.br1 = nn.BatchNorm2d(self.ndf*2)
        self.cr2 = nn.Conv2d(self.ndf*2, self.ndf * 2, 3, 1, 0)
        self.br2 = nn.BatchNorm2d(self.ndf * 2)

        #res block 2
        self.cr3 = nn.Conv2d(self.ndf*2, self.ndf*2, 3, 2, 0)
        self.br3 = nn.BatchNorm2d(self.ndf*2)
        self.cr4 = nn.Conv2d(self.ndf*2, self.ndf * 2, 3, 1, 0)
        self.br4 = nn.BatchNorm2d(self.ndf * 2)

        self.e4 = nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.ndf * 4)
        self.e5 = nn.Conv2d(self.ndf * 4, self.ndf * 4, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(self.ndf * 4)
        self.fc1 = nn.Linear(self.ndf * 8 * 2, latent_variable_size)
        self.leakyrelu = nn.LeakyReLU(0.2) #  inplace=True

        #self.decoder = nn.Sequential(
        self.tc1 = nn.ConvTranspose2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.c1 = nn.ConvTranspose2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.b1 = nn.BatchNorm2d(512)


        self.tc2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.c2 = nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.b2 = nn.BatchNorm2d(256)

        #res block 1
        self.dcr1 = nn.Conv2d(128, 128, (3,3), 2, 0)
        self.dbr1 = nn.BatchNorm2d(128)
        self.dcr2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.dbr2 = nn.BatchNorm2d(128)

        #res block 2
        self.dcr3 = nn.Conv2d(128, 128, 3, 2, 0)
        self.dbr3 = nn.BatchNorm2d(128)
        self.dcr4 = nn.Conv2d(128, 128, 3, 1, 0)
        self.dbr4 = nn.BatchNorm2d(128)


        self.tc3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.c3 = nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.b3 = nn.BatchNorm2d(128)
        self.tc4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.c4 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.b4 = nn.BatchNorm2d(64)

        #res block 3
        self.dcr5 = nn.Conv2d(64, 64, (3,3), 1, 1)
        self.dbr5 = nn.BatchNorm2d(64)
        self.dcr6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dbr6 = nn.BatchNorm2d(64)

        #res block 4
        self.dcr7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dbr7 = nn.BatchNorm2d(64)
        self.dcr8 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dbr8 = nn.BatchNorm2d(64)


        self.tc5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.c5 = nn.ConvTranspose2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.b5 = nn.BatchNorm2d(32)
        self.tc6 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.b6 = nn.BatchNorm2d(32)
        self.tc7 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.b7 = nn.BatchNorm2d(3)
        self.tanh = nn.Tanh()

    def encode(self, x):

        h0 = self.leakyrelu(self.bn0(self.e0(x)))
        h1 = self.leakyrelu(self.bn1(self.e1(h0)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))

        #first residual block
        r1 = self.leakyrelu(self.br1(self.cr1(h3)))
        r2 = self.br2(self.cr2(r1))
        r3 = r2 + h3
        x3 = self.leakyrelu(r3)

        #second residual block
        r4 = self.leakyrelu(self.br3(self.cr3(x3)))
        r5 = self.br3(self.cr3(r4))
        r5 = r3 + r5
        r6 = self.leakyrelu(r5)

        h4 = self.leakyrelu(self.bn4(self.e4(r6)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*2)
        return self.fc1(h5)

    def decode(self, z):

        h1 = self.leakyrelu(self.b1(self.tc1(z)))
        h2 = self.leakyrelu(self.b2(self.tc2(h1)))
        h3 = self.leakyrelu(self.b3(self.tc3(h2)))

        #first residual block
        r1 = self.leakyrelu(self.dbr1(self.dcr1(h3)))
        r2 = self.dbr2(self.dcr2(r1))
        r3 = r2 + h3
        x3 = self.leakyrelu(r3)

        #second residual block
        r4 = self.leakyrelu(self.dbr3(self.dcr3(x3)))
        r5 = self.dbr4(self.dcr4(r4))
        r5 = r3 + r5
        r6 = self.leakyrelu(r5)
        h4 = self.leakyrelu(self.b4(self.tc4(r6)))

        #third residual block
        r11 = self.leakyrelu(self.dbr5(self.dcr5(h4)))
        r21 = self.dbr6(self.dcr6(r11))
        r31 = r21 + h4
        x31 = self.leakyrelu(r31)

        #forth residual block
        r41 = self.leakyrelu(self.dbr7(self.dcr7(x31)))
        r51 = self.dbr8(self.dcr8(r41))
        r51 = r31 + r51
        r61 = self.leakyrelu(r51)


        h5 = self.leakyrelu(self.b5(self.tc5(r61)))
        h6 = self.leakyrelu(self.b6(self.tc6(h5)))
        h7 = self.leakyrelu(self.b7(self.tc7(h6)))
        return self.tanh(h7)

    def forward(self, x):
        encoded = self.encode(x)
        encoded = encoded.view(encoded.size()[0], 256, 1, 1)
        decoded = self.decode(encoded)
        return encoded, decoded


if (IS_GPU == 1):
    model = AutoEncoder().cuda()
    if (LOAD_WEIGHTS == 1):
        model.load_state_dict(torch.load(AE_WEIGHTS))
        #stop updating the encoder
        for p in model.e1.parameters():
            p.requires_grad = False
        for p in model.e2.parameters():
            p.requires_grad = False
        for p in model.e3.parameters():
            p.requires_grad = False
        for p in model.e4.parameters():
            p.requires_grad = False
        for p in model.e5.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.bn2.parameters():
            p.requires_grad = False
        for p in model.bn3.parameters():
            p.requires_grad = False
        for p in model.bn4.parameters():
            p.requires_grad = False
        for p in model.bn5.parameters():
            p.requires_grad = False
        for p in model.fc1.parameters():
            p.requires_grad = False
        for p in model.cr1.parameters():
            p.requires_grad = False
        for p in model.cr2.parameters():
            p.requires_grad = False
        for p in model.cr3.parameters():
            p.requires_grad = False
        for p in model.cr4.parameters():
            p.requires_grad = False
        for p in model.br1.parameters():
            p.requires_grad = False
        for p in model.br2.parameters():
            p.requires_grad = False
        for p in model.br3.parameters():
            p.requires_grad = False
        for p in model.br4.parameters():
            p.requires_grad = False
else:
    model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

for epoch in range(EPOCH):
    start = time.time()
    for data in train_loader:
        img = data
        if (IS_GPU == 1):
            img = Variable(img).cuda()
        else:
            img = Variable(img)
        img = img.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output[1], img)
        # ===================backward====================
        optimizer.zero_grad() #make sure all gradients are zero before backprop
        loss.backward()
        optimizer.step()
    # ===================log========================
    end = time.time()
    print('epoch [{}/{}], loss:{:.4f}, time:{}'
          .format(epoch + 1, EPOCH, loss.data[0], end - start))
    sys.stdout.flush()
    if ((epoch % 10 == 9) or (epoch > 400)):
        img = img / 2 + 0.5  # unnormalize
        save = img.view(img.size(0), CHANNELS, HEIGHT, WIDTH)
        save_image(save, './input_{}.png'.format(epoch))
        temp = output[1] / 2 + 0.5
        save = temp.view(temp.size(0), CHANNELS, HEIGHT, WIDTH)
        save_image(save, './output_{}.png'.format(epoch))
        for data in test_loader:
            img = data
            if(IS_GPU == 1):
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            img = img.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
            output = model(img)
            img = img / 2 + 0.5  # unnormalize
            save = img.view(img.size(0), CHANNELS, HEIGHT, WIDTH)
            save_image(save, SAVE_RESULTS_DIR + 'test_input_{}.png'.format(epoch))
            temp = output[1] / 2 + 0.5
            save = temp.view(temp.size(0), CHANNELS, HEIGHT, WIDTH)
            save_image(save, SAVE_RESULTS_DIR + 'test_output_{}.png'.format(epoch))
            break

torch.save(model.state_dict(), SAVE_AE_WEIGHTS)

