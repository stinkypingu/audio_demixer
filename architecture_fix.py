import torch
import torch.optim as optim
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, i, o):
        super(EncoderBlock, self).__init__()

        #first convolution
        self.conv2d_1 = nn.Conv2d(i, o, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_1 = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm2d(o)
        
        #second convolution
        self.conv2d_2 = nn.Conv2d(o, o, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_2 = nn.ReLU()
        self.batch_norm_2 = nn.BatchNorm2d(o)

        #reduce dimensionality
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        #first convolution
        x = self.conv2d_1(x)
        x = self.activation_1(x)
        x = self.batch_norm_1(x)
        
        #second convolution
        x = self.conv2d_2(x)
        x = self.activation_2(x)
        x = self.batch_norm_2(x)

        #reduce dimensionality
        x_pool = self.max_pool(x)
        
        return x, x_pool
    


        
class DecoderBlock(nn.Module):
    def __init__(self, i, o):
        super(DecoderBlock, self).__init__()

        #upsample to double the resolution here
        self.convT_1 = nn.ConvTranspose2d(i, o, kernel_size=4, stride=2, padding=1, bias=False)
        self.activation_1 = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm2d(o)

        #dropout to prevent overfitting
        self.dropout = nn.Dropout2d(0.4)

        #first deconvolution, need to handle new input shape from concatenation
        self.convT_2 = nn.ConvTranspose2d(2 * o, o, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_2 = nn.ReLU()
        self.batch_norm_2 = nn.BatchNorm2d(o)

        #second deconvolution
        self.convT_3 = nn.ConvTranspose2d(o, o, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_3 = nn.ReLU()
        self.batch_norm_3 = nn.BatchNorm2d(o)

    def forward(self, x, skip_connection):
        x = self.convT_1(x)
        x = self.activation_1(x)
        x = self.batch_norm_1(x)
        
        #concatenation
        x = torch.cat([x, skip_connection], dim=1)

        x = self.convT_2(x)
        x = self.activation_2(x)
        x = self.batch_norm_2(x)

        x = self.convT_3(x)
        x = self.activation_3(x)
        x = self.batch_norm_3(x)

        return x



class UNet(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        
        self.encoder1 = EncoderBlock(i, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)
        self.encoder4 = EncoderBlock(128, 256)

        self.center_conv = EncoderBlock(256, 512)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)

        self.final_conv = nn.Conv2d(32, o, kernel_size=1)

    def forward(self, x):
        x1, x_pool1 = self.encoder1(x)
        #print(f'encoder1, x1:\t\t{x1.shape}')
        #print(f'encoder1, x_pool1:\t{x_pool1.shape}\n')

        x2, x_pool2 = self.encoder2(x_pool1)
        #print(f'encoder2, x2:\t\t{x2.shape}')
        #print(f'encoder2, x_pool2:\t{x_pool2.shape}\n')

        x3, x_pool3 = self.encoder3(x_pool2)
        x4, x_pool4 = self.encoder4(x_pool3)
        #print(f'encoder3, x3:\t\t{x3.shape}')
        #print(f'encoder3, x_pool3:\t{x_pool3.shape}\n')


        center, _ = self.center_conv(x_pool4)
        #print(f'center:\t\t\t{center.shape}\n')

        x = self.decoder4(center, x4)
        x = self.decoder3(x, x3)
        #print(f'decoder3, x:\t{x.shape}\n')

        x = self.decoder2(x, x2)
        #print(f'decoder2, x:\t{x.shape}\n')

        x = self.decoder1(x, x1)
        #print(f'decoder1, x:\t{x.shape}\n')

        x = self.final_conv(x)

        return x


