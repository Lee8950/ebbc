import cbam
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data.dataloader
import pickle
import torch
import torch.utils.data
import PIL.Image
import os
import sys
import torchvision.transforms


class StackedLayers(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=0.01, kernel_size=(3,3), padding=1, leaky=0.01):
        super(StackedLayers, self).__init__()
        self.cbam = cbam.CBAM(in_channels, reduction_ratio, kernel_size=3)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", device="cuda", dtype=torch.float32)
        self.bn = nn.BatchNorm2d(out_channels, device="cuda", dtype=torch.float32)
        self.leaky = nn.LeakyReLU(negative_slope=leaky)
    
    def forward(self, x):
        # Resnet
        y = self.cbam(x) + x
        y = self.conv(y)
        y = self.leaky(y)
        return y

class EBBCEncoder(nn.Module):
    def __init__(self, leaky=0.01):
        super(EBBCEncoder, self).__init__()
        self.layers = nn.Sequential(StackedLayers(3, 64, leaky),
                                    nn.BatchNorm2d(64, device="cuda", dtype=torch.float32),
                                    nn.MaxPool2d((2,2)),
                                    StackedLayers(64, 64, leaky),
                                    nn.BatchNorm2d(64, device="cuda", dtype=torch.float32),
                                    nn.MaxPool2d((2,2)),
                                    
                                    StackedLayers(64, 64, leaky),
                                    nn.BatchNorm2d(64, device="cuda", dtype=torch.float32),
                                    StackedLayers(64, 64, leaky),
                                    nn.BatchNorm2d(64, device="cuda"),
                                    
                                    StackedLayers(64, 128, leaky),
                                    nn.BatchNorm2d(128, device="cuda", dtype=torch.float32),
                                    nn.MaxPool2d((2,2)),
                                    
                                    StackedLayers(128, 128, leaky),
                                    nn.BatchNorm2d(128, device="cuda", dtype=torch.float32),
                                    StackedLayers(128, 128, leaky),
                                    nn.BatchNorm2d(128, device="cuda"),
                                    
                                    StackedLayers(128, 256, leaky),
                                    nn.BatchNorm2d(256, device="cuda", dtype=torch.float32),
                                    nn.MaxPool2d((2,2)),
                                    
                                    StackedLayers(256, 256, leaky),
                                    nn.BatchNorm2d(256, device="cuda", dtype=torch.float32),
                                    StackedLayers(256, 256, leaky),
                                    nn.BatchNorm2d(256),
                                    StackedLayers(256, 256, leaky),
                                    nn.BatchNorm2d(256),
                                    StackedLayers(256, 256, leaky),
                                    nn.BatchNorm2d(256),
                                    
                                    StackedLayers(256, 512, leaky),
                                    nn.BatchNorm2d(512, device="cuda", dtype=torch.float32),
                                    nn.MaxPool2d((2,2)),
                                    
                                    StackedLayers(512, 1024, leaky),
                                    nn.BatchNorm2d(1024, device="cuda", dtype=torch.float32),
                                    StackedLayers(1024, 1024, leaky),
                                    nn.BatchNorm2d(1024, device="cuda", dtype=torch.float32),
                                    StackedLayers(1024, 1024, leaky),
                                    nn.BatchNorm2d(1024),
                                    StackedLayers(1024, 1024, leaky),
                                    nn.BatchNorm2d(1024),
                                    StackedLayers(1024, 1024, leaky),
                                    nn.BatchNorm2d(1024),
                                    
                                    nn.Conv2d(1024, 1000, (3,3), padding="same", device="cuda", dtype=torch.float32),
                                    nn.AdaptiveAvgPool2d((1,1))
        )
    
    def forward(self, x):
        return self.layers(x)

class DecoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, pic_size, reduction_ratio=0.01, kernel_size=(3,3), padding=1):
        super(DecoderConv, self).__init__()
        self.cbam1 = cbam.CBAM(in_channels, reduction_ratio, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding="same", device="cuda", dtype=torch.float32)
        self.avg = nn.AdaptiveAvgPool2d(pic_size)#nn.AvgPool2d(kernel_size=2) 
        self.maxpool = nn.AdaptiveMaxPool2d(pic_size)#nn.MaxPool2d(kernel_size=(2,2))
        self.sigmoid = nn.Sigmoid()
        self.cbam2 = cbam.CBAM(in_channels, reduction_ratio, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", device="cuda", dtype=torch.float32)
        
    def forward(self, x):
        m = self.avg(x)
        
        m = self.maxpool(m)
        m = self.conv1(m)
        m = self.sigmoid(m)
        m = self.cbam1(m)
        #print(x.shape, m.shape)
        y = x + m
        y = self.cbam2(y)
        y = self.conv2(y)
        return y

class EBBCDecoder(nn.Module):
    def __init__(self):
        super(EBBCDecoder, self).__init__()
        self.layers = nn.Sequential(nn.Upsample(scale_factor=8),
                                    DecoderConv(1000, 1024, (8,8)),
                                    nn.BatchNorm2d(1024, device="cuda"),
                                    
                                    DecoderConv(1024, 1024, (8,8)),
                                    nn.BatchNorm2d(1024),
                                    DecoderConv(1024, 1024, (8,8)),
                                    nn.BatchNorm2d(1024),
                                    DecoderConv(1024, 1024, (8,8)),
                                    nn.BatchNorm2d(1024),
                                    DecoderConv(1024, 1024, (8,8)),
                                    nn.BatchNorm2d(1024, device="cuda"),
                                    
                                    nn.Upsample(scale_factor=2),
                                    DecoderConv(1024, 512, (16,16)),
                                    nn.BatchNorm2d(512, device="cuda"),
                                    
                                    DecoderConv(512, 512, (16,16)),
                                    nn.BatchNorm2d(512),
                                    DecoderConv(512, 512, (16,16)),
                                    nn.BatchNorm2d(512),
                                    DecoderConv(512, 512, (16,16)),
                                    nn.BatchNorm2d(512),
                                    DecoderConv(512, 512, (16,16)),
                                    nn.BatchNorm2d(512, device="cuda"),
                                    
                                    nn.Upsample(scale_factor=2),
                                    DecoderConv(512, 256, (32,32)),
                                    nn.BatchNorm2d(256, device="cuda"),
                                    
                                    DecoderConv(256, 256, (32,32)),
                                    nn.BatchNorm2d(256, device="cuda"),
                                    DecoderConv(256, 256, (32,32)),
                                    nn.BatchNorm2d(256),
                                    DecoderConv(256, 256, (32,32)),
                                    nn.BatchNorm2d(256),
                                    
                                    nn.Upsample(scale_factor=2),
                                    DecoderConv(256, 128, (64,64)),
                                    nn.BatchNorm2d(128, device="cuda"),
                                    
                                    DecoderConv(128, 128, (64,64)),
                                    nn.BatchNorm2d(128, device="cuda"),
                                    DecoderConv(128, 128, (64,64)),
                                    nn.BatchNorm2d(128),
                                    
                                    nn.Upsample(scale_factor=2),
                                    DecoderConv(128, 64, (128,128)),
                                    nn.BatchNorm2d(64, device="cuda"),
                                    
                                    nn.Upsample(scale_factor=2),
                                    DecoderConv(64, 3, (256,256)),
                                    nn.BatchNorm2d(3, device="cuda"),
        )
    
    def forward(self, x):
        return self.layers(x)
    
class TrainData(torch.utils.data.Dataset):
    def __init__(self, loc):
        imgs_names = os.listdir(f"{loc}/style_images/")
        image_files = [f for f in imgs_names if f.lower().endswith(("jpg"))]
        answ_names = os.listdir(f"{loc}/original_images/")
        answ_files = [f for f in answ_names if f.lower().endswith(("jpg"))]
        
        self.imgs = []
        self.answ = []
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32)])
        
        for image_file in image_files:
            with PIL.Image.open(f"{loc}/style_images/{image_file}") as fp:
                self.imgs.append(transform(fp).unsqueeze(0).cuda())
                
        for answ_file in answ_files:
            with PIL.Image.open(f"{loc}/original_images/{answ_file}") as fp:
                self.answ.append(transform(fp).unsqueeze(0).cuda())

        #print(self.imgs, self.answ)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.answ[idx]

class TestData(torch.utils.data.Dataset):
    def __init__(self, loc):
        imgs_names = os.listdir(f"{loc}/style_images/")
        image_files = sorted([f for f in imgs_names if f.lower().endswith(("jpg"))])
        answ_names = os.listdir(f"{loc}/original_images/")
        answ_files = sorted([f for f in answ_names if f.lower().endswith(("jpg"))])
        
        self.imgs = []
        self.answ = []
        
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float32)])
        
        for image_file in image_files:
            with PIL.Image.open(f"{loc}/style_images/{image_file}") as fp:
                pic = transform(fp).unsqueeze(0)
                self.imgs.append(pic.cuda())
                
        for answ_file in answ_files:
            with PIL.Image.open(f"{loc}/original_images/{answ_file}") as fp:
                pic = transform(fp).unsqueeze(0)
                self.answ.append(pic.cuda())

        print(imgs_names, answ_names)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.answ[idx]

class EBBC(nn.Module):
    def __init__(self):
        super(EBBC, self).__init__()
        self.encoder = EBBCEncoder()
        self.decoder = EBBCDecoder()
    
    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        return y

def main():
    debug = False
    net = EBBC()
    net.to(torch.device("cuda"))
    train_enable = True
    
    loc = os.path.dirname(__file__)
    
    train_data_loader = TrainData(loc)
    test_data_loader = TestData(loc)
    
    optimizer = torch.optim.Adam(net.parameters(), 0.01)
    
    if train_enable:
        print(f"Training Start, step:{len(train_data_loader)}")
        for _ in range(25):
            epoch_msg = f"Epoch {_+1}/{25}"
            steps = len(train_data_loader)
            for idx, (img, ans) in enumerate(train_data_loader):
                img_gpu, ans_gpu = img, ans
                #print(img_gpu.shape, ans_gpu.shape)
                res = net(img_gpu)
                loss = torch.mean((res - ans_gpu) ** 2)
                loss_val = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Training:{epoch_msg}:Step:{idx}/{steps} loss:{loss_val}                            ",end='\r')
                if idx % 4 == 0:
                    torch.save(net, "model.pth")
                    torch.save(net.state_dict(), 'model_params.pth')
    

    
if __name__ == "__main__":
    main()