
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
import random

#Folders used #TODO - update those folders
MUSTACHE_DIR = "/home/student4/link/train_m/" #Folder contains mustache images for training
NO_MUSTACHE_TRAIN_DIR = "/home/student4/link/no_beard_training/" #Folder contains mustacheless images for training
NO_MUSTACHE_TEST_DIR = "/home/student4/link/no_beard_testing/" #Folder contains mustachless images for testing
SAVE_RESULTS_DIR = "/home/student4/link/ae8_results/" #Where you want to save your results
AE_WEIGHTS =  "/home/student4/Eliran/new_autoencoder128.pth" #weights of already trained autoencoder
SAVE_AE_WEIGHTS = "/home/student4/link/ae8_results/new_autoencoder.pth" #saving generator weights
SAVE_D_WEIGHTS =   "/home/student4/link/ae8_results/new_discriminator.pth" #saving discriminator weights
#D_WEIGHTS = "/home/student4/Eliran/our_discriminator.pth"

# Hyper Parameters
ALPHA = 100         #As defined in the originial article
BETTA = 100
GAMMA = 0
EPOCH = 250         #number of epochs
BATCH_SIZE = 64
NOISE = 0.01         #initial noise
G_LR = 0.0001        #G learning rate
D_LR = 0.0001        #D learning rate
N_MUSTACHE_IMG = BATCH_SIZE * 121 #Number of Train images
N_NO_MUSTACHE_TRAIN_IMG = BATCH_SIZE * 121 #Number of Train images
N_NO_MUSTACHE_TEST_IMG = BATCH_SIZE * 1 #Number of Test images
HEIGHT = 128 #Dimensions of single image
WIDTH = 128
CHANNELS = 3 #RGB
IMG_SIZE = HEIGHT * WIDTH * CHANNELS
IS_GPU = 1 #Please note the code only tested when IS_GPU=1
LOAD_WEIGHTS = 1 #Load the weights of the autoencoder

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def gaussian_noise(inputs, mean=0, stddev=0.01):
    input = inputs.cpu()
    input_array = input.data.numpy()

    noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(input_array))

    out = np.add(input_array, noise)

    output_tensor = torch.from_numpy(out)
    out_tensor = Variable(output_tensor)
    out = out_tensor.cuda()
    out = out.float()
    return out

#Helper function fot Ltv
def g_train_smoothing_functions(s_G):
    B, C, H, W = s_G.size()

    gen_ten = s_G.contiguous().view(B, C, H, W)
    z_ijp1 = gen_ten[:, :, 1:, :-1]
    z_ijv1 = gen_ten[:, :, :-1, :-1]
    z_ip1j = gen_ten[:, :, :-1, 1:]
    z_ijv2 = gen_ten[:, :, :-1, :-1]

    diff1 = z_ijp1 - z_ijv1
    diff1 = torch.abs(diff1)
    diff2 = z_ip1j - z_ijv2
    diff2 = torch.abs(diff2)

    diff_sum = diff1 + diff2
    per_chan_avg = torch.sum(diff_sum, dim=1)
    loss = torch.sum(torch.sum(per_chan_avg, dim=1), dim=1)
    loss = torch.sum(loss)
    loss = loss / (CHANNELS * BATCH_SIZE)
    return loss

#DataSet class
class Faces(Data.Dataset):
    """Faces."""

    def __init__(self, root_dir, transform, size):

        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.size = size #TODO

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

mustache_data = Faces(MUSTACHE_DIR, transform, N_MUSTACHE_IMG)
train_data = Faces(NO_MUSTACHE_TRAIN_DIR, transform, N_NO_MUSTACHE_TRAIN_IMG)
test_data = Faces(NO_MUSTACHE_TEST_DIR, transform, N_NO_MUSTACHE_TEST_IMG)

#Create the data loaders
mustache_loader = Data.DataLoader(dataset=mustache_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


class D_NET(nn.Module):
    def __init__(self):
        super(D_NET, self).__init__()
        self.ndf = 2
        self.e1 = nn.Conv2d(3, self.ndf, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(self.ndf)
        self.r1 = nn.Conv2d(3, self.ndf*2, 5, 1, 2)
        self.bnr1 = nn.BatchNorm2d(self.ndf*2)
        self.e2 = nn.Conv2d(self.ndf*2, self.ndf * 4, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.ndf * 4)
        self.r2 = nn.Conv2d(self.ndf * 2, self.ndf * 4, 5, 1, 2)
        self.bnr2 = nn.BatchNorm2d(self.ndf * 4)
        self.e3 = nn.Conv2d(self.ndf * 8, self.ndf * 16, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.ndf * 16)
        self.r3 = nn.Conv2d(self.ndf * 4, self.ndf * 8, 5, 1, 2)
        self.bnr3 = nn.BatchNorm2d(self.ndf * 8)
        self.e4 = nn.Conv2d(self.ndf * 32, self.ndf * 64, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.ndf * 64)
        self.r4 = nn.Conv2d(self.ndf * 8, self.ndf * 16, 5, 1, 2)
        self.bnr4 = nn.BatchNorm2d(self.ndf * 16)
        self.e5 = nn.Conv2d(self.ndf * 128, self.ndf * 128, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(self.ndf * 128)
        self.r5 = nn.Conv2d(self.ndf * 16, self.ndf * 32, 5, 1, 2)
        self.bnr5 = nn.BatchNorm2d(self.ndf * 32)
        self.e6 = nn.Conv2d(self.ndf * 32, self.ndf * 16, 3, 2, 1)
        self.bn6 = nn.BatchNorm2d(self.ndf * 16)
        self.e7 = nn.Conv2d(self.ndf * 16, self.ndf, 3, 2, 1)
        self.bn7 = nn.BatchNorm2d(self.ndf)
        self.fc1 = nn.Linear(self.ndf * 16 * 64, 3)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f1 = self.leakyrelu(self.bnr1(self.r1(x)))
        f2 = self.leakyrelu(self.bnr2(self.r2(f1)))
        f3 = self.leakyrelu(self.bnr3(self.r3(f2)))
        f4 = self.leakyrelu(self.bnr4(self.r4(f3)))
        f5 = self.leakyrelu(self.bnr5(self.r5(f4)))
        h6 = self.leakyrelu(self.bn6(self.e6(f5)))
        h7 = self.leakyrelu(self.bn7(self.e7(h6)))
        h7 = h7.view(-1, self.ndf * 16 * 64)
        h8 = self.sigmoid(self.fc1(h7))
        h8 = h8.view(-1,3)
        return h8


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
    ae = AutoEncoder().cuda()
    d_net = D_NET().cuda()
    if (LOAD_WEIGHTS == 1):
        #d_net.load_state_dict(torch.load(D_WEIGHTS))
        d_net.apply(weights_init)
        ae.load_state_dict(torch.load(AE_WEIGHTS))
        #stop updating the encoder
        for p in ae.e0.parameters():
            p.requires_grad = False
        for p in ae.e1.parameters():
            p.requires_grad = False
        for p in ae.e2.parameters():
            p.requires_grad = False
        for p in ae.e3.parameters():
            p.requires_grad = False
        for p in ae.e4.parameters():
            p.requires_grad = False
        for p in ae.e5.parameters():
            p.requires_grad = False
        for p in ae.bn0.parameters():
            p.requires_grad = False
        for p in ae.bn1.parameters():
            p.requires_grad = False
        for p in ae.bn2.parameters():
            p.requires_grad = False
        for p in ae.bn3.parameters():
            p.requires_grad = False
        for p in ae.bn4.parameters():
            p.requires_grad = False
        for p in ae.bn5.parameters():
            p.requires_grad = False
        for p in ae.fc1.parameters():
            p.requires_grad = False
        for p in ae.cr1.parameters():
            p.requires_grad = False
        for p in ae.cr2.parameters():
            p.requires_grad = False
        for p in ae.cr3.parameters():
            p.requires_grad = False
        for p in ae.cr4.parameters():
            p.requires_grad = False
        for p in ae.br1.parameters():
            p.requires_grad = False
        for p in ae.br2.parameters():
            p.requires_grad = False
        for p in ae.br3.parameters():
            p.requires_grad = False
        for p in ae.br4.parameters():
            p.requires_grad = False
else:
    ae = AutoEncoder()


#Loss criterion
mse_criterion = nn.MSELoss()
cel_criterion = nn.CrossEntropyLoss().cuda()
bce_criterion = nn.BCELoss().cuda()

#Optimizer
g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), lr=G_LR)
d_optimizer = torch.optim.Adam(d_net.parameters(), lr=D_LR)

#Label to compare to in loss criterion
label_0, label_1, label_2 = (torch.LongTensor(BATCH_SIZE) for i in range(3))
if (IS_GPU == 1):
    label_0 = Variable(label_0.cuda())
    label_1 = Variable(label_1.cuda())
    label_2 = Variable(label_2.cuda())
else:
    label_0 = Variable(label_0)
    label_1 = Variable(label_1)
    label_2 = Variable(label_2)
label_0.data.resize_(BATCH_SIZE).fill_(0)
label_1.data.resize_(BATCH_SIZE).fill_(1)
label_2.data.resize_(BATCH_SIZE).fill_(2)


#init veriable
train_d_flag = True
noise_decay = NOISE / EPOCH
current_noise = NOISE

#test img
test_iter = iter(test_loader)
test_img = test_iter.next()
test_img = Variable(test_img).cuda()
test_img = test_img.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)


for epoch in range(EPOCH):
    current_noise = max(current_noise - noise_decay, 0.001)
    start = time.time()

    train_iter = iter(train_loader)
    mustache_iter = iter(mustache_loader)
    avg_d = 0 #average d loss
    avg_g = 0 #average g loss
    d_cnt = 0
    g_cnt = 0

    #number of images in each epoch
    l = min(len(train_iter), len(mustache_iter))

    for i in range(l): #for each batch
        m_img = mustache_iter.next() #mustache
        t_img = train_iter.next() #without mustache

        # with probability 0.5 flip horizontally
        for j in range(0,BATCH_SIZE):
            coin = random.random()
            if coin > 0.5 :
                m_img[j] = m_img[j] / 2 + 0.5
                p_transform = transforms.ToPILImage()
                i_transform = transforms.ToTensor()
                n_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                temp = m_img[j]
                temp = p_transform(temp)
                temp = transforms.functional.hflip(temp)
                temp = i_transform(temp)
                temp = n_transform(temp)
                m_img[j] = temp

        for j in range(0,BATCH_SIZE):
            coin = random.random()
            if coin > 0.5 :
                t_img[j] = t_img[j] / 2 + 0.5
                p_transform = transforms.ToPILImage()
                i_transform = transforms.ToTensor()
                n_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                temp = t_img[j]
                temp = p_transform(temp)
                temp = transforms.functional.hflip(temp)
                temp = i_transform(temp)
                temp = n_transform(temp)
                t_img[j] = temp

        if (IS_GPU == 1):
            m_img = Variable(m_img).cuda()
            t_img = Variable(t_img).cuda()
        else:
            m_img = Variable(m_img)
            t_img = Variable(t_img)

        m_img = m_img.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        t_img = t_img.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

        m_img  = gaussian_noise(m_img, mean=0, stddev=current_noise)
        t_img = gaussian_noise(t_img, mean=0, stddev=current_noise)

        if i % 100 == 0: #Train the discriminator one batch for 100 batches of generator training
            #---=== TRAIN D ===---#
            d_net.train()
            ae.eval()
            d_optimizer.zero_grad()
            d3 = d_net(m_img)
            loss_d3 = cel_criterion(d3, label_2)
            m_img_after_ae = ae(m_img)[1]
            d2 = d_net(m_img_after_ae.detach())
            loss_d2 = cel_criterion(d2, label_1)
            t_img_after_ae = ae(t_img)[1]
            d1 = d_net(t_img_after_ae.detach())
            loss_d1 = cel_criterion(d1, label_0)

            d_loss = loss_d1 + loss_d2 + loss_d3

            d_loss.backward()
            d_optimizer.step()


        if 1 == 1:
            #---=== TRAIN G ===---#
            d_net.eval()
            ae.train()
            g_optimizer.zero_grad()

            # LTID
            ae_out_m = ae(m_img)[1]
            Ltid = mse_criterion(ae_out_m, m_img)

            # Lconst
            after_ae = ae(t_img)
            f_t = after_ae[0]  # f(x)
            g_f_t = after_ae[1]  # g(f(x))
            f_g_f_t = ae(g_f_t)[0]  # f(g(f(x)))
            f_g_f_t = f_g_f_t
            Lconst = mse_criterion(f_g_f_t, f_t)

            # LGANG
            m_img_after_ae = ae(m_img)[1]
            d2 = d_net(m_img_after_ae)
            loss_d2 = cel_criterion(d2, label_2)  # FIXME
            t_img_after_ae = ae(t_img)[1]
            d1 = d_net(t_img_after_ae)
            loss_d1 = cel_criterion(d1, label_2)  # FIXME
            Lgang = loss_d1 + loss_d2

            # Ltv
            # Ltv = g_train_smoothing_functions(m_img_after_ae) + g_train_smoothing_functions(t_img_after_ae);
            Ltv = 0  # Ltv / 2

            g_loss = Lgang + ALPHA * Lconst + BETTA * Ltid + GAMMA * Ltv

            g_loss.backward()
            g_optimizer.step()
            avg_g = avg_g + float(g_loss[0])
            g_cnt = g_cnt + 1


    # ===================log========================
    end = time.time()
    print("=============")
    try:
        print('epoch [{}/{}], d_loss:{:.4f}, g_loss:{:.4f}, time:{}'
          .format(epoch + 1, EPOCH, d_loss, g_loss, end - start))
    except NameError:
        print('epoch [{}/{}], d_loss:{:.4f}, time:{}'
              .format(epoch + 1, EPOCH, d_loss, end - start))
        print("noise" + str(current_noise))
    print("=============")

    sys.stdout.flush()

    #Run and save tests every 5 epochs
    if (epoch % 5 == 4):
        for data in test_loader:
            img = data
            if(IS_GPU == 1):
                img = Variable(img).cuda()
            else:
                img = Variable(img)
            img = img.view(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
            output = ae(img)
            img = img / 2 + 0.5  # unnormalize
            save = img.view(img.size(0), CHANNELS, HEIGHT, WIDTH)
            save_image(save, SAVE_RESULTS_DIR + 'test_input_{}.png'.format(epoch))
            temp = output[1] / 2 + 0.5
            save = temp.view(temp.size(0), CHANNELS, HEIGHT, WIDTH)
            save_image(save, SAVE_RESULTS_DIR + 'test_output_{}.png'.format(epoch))
            break

#Save net's weights at the end of the run
torch.save(ae.state_dict(), SAVE_AE_WEIGHTS)
torch.save(d_net.state_dict(), SAVE_D_WEIGHTS)

