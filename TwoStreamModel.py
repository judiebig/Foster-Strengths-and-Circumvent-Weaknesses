import torch
import torch.nn as nn
import os
import numpy as np
# from tools.complex_tools import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TwoStreamModel(nn.Module):
    def __init__(self):
        super(TwoStreamModel, self).__init__()
        # model 1 for amplitude
        self.en_1 = Encoder()
        self.de_1 = Decoder()
        self.TCMs_1 = nn.Sequential(TCM(),
                                    TCM(),
                                    TCM())
        self.lin_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1,kernel_size=1,stride=1),
                                   nn.Softplus())
        # model 2 for complex
        self.en_2 = Encoder()
        self.de_2_real = Decoder()
        self.de_2_imag = Decoder()
        self.TCMs_2 = nn.Sequential(TCM(),
                                    TCM(),
                                    TCM())


    def forward(self, x):
        # x = x.unsqueeze(dim=1)  # complex input
        ori = x
        # model 1
        x, en_list_1, Com = self.en_1(ori)  # [b,c,t,f]
        x = x.permute(0, 2, 1, 3)  # [b,t,c,f]
        x = x.reshape(x.size()[0], x.size()[1], -1).permute(0, 2, 1)  # [b, cf, t]
        x = self.TCMs_1(x).permute(0, 2, 1)
        x = x.reshape(x.size()[0], x.size()[1], 64, 4)  # [b,t,c,f]
        x = x.permute(0, 2, 1, 3)
        x, de_list_1 = self.de_1(x, en_list_1)
        Amp = self.lin_1(x)

        # model 2
        x, en_list_2 = self.en_2(ori, Com)  # [b,c,t,f]
        x = x.permute(0, 2, 1, 3)  # [b,t,c,f]
        x = x.reshape(x.size()[0], x.size()[1], -1).permute(0, 2, 1)  # [b, cf, t]
        x = self.TCMs_2(x).permute(0, 2, 1)
        x = x.reshape(x.size()[0], x.size()[1], 64, 4)  # [b,t,c,f]
        x = x.permute(0, 2, 1, 3)
        x_real, dereal_list_2 = self.de_2_real(x, en_list_2)
        x_imag, deimag_list_2 = self.de_2_imag(x, en_list_2)
        # x_2 = self.lin_2(x)
        Complex = torch.cat((x_real, x_imag),dim=1)
        del en_list_1, en_list_2, de_list_1, dereal_list_2, deimag_list_2
        return Amp.squeeze(), Complex


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)  # left right up down

        # convGLU
        self.conv1 = BiConvGLU(in_channels=2, out_channels=64, kernel_size=(2, 5), stride=(1, 2))
        self.conv2 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv3 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv4 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv5 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.en1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
            )
        self.en2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.en3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.en4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.en5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

    def forward(self, x, de_list=None):
        en_list = []
        if de_list !=None:
            # print(x.shape)
            x = self.pad1(x)
            x = self.conv1(x,de_list[0])
            x = self.en1(x)
            en_list.append(x)
            x = self.pad1(x)
            x = self.conv2(x,de_list[1])
            x = self.en2(x)
            en_list.append(x)
            x = self.pad1(x)
            x = self.conv3(x,de_list[2])
            x = self.en3(x)
            en_list.append(x)
            x = self.pad1(x)
            x = self.conv4(x,de_list[3])
            x = self.en4(x)
            en_list.append(x)
            x = self.pad1(x)
            x = self.conv5(x,de_list[4])
            x = self.en5(x)
            en_list.append(x)
        else:
            Com = []
            x = self.pad1(x)
            _,x = self.conv1(x)
            x = self.en1(x)
            en_list.append(x)
            Com.append(_)
            x = self.pad1(x)
            _,x = self.conv2(x)
            x = self.en2(x)
            Com.append(_)
            en_list.append(x)
            x = self.pad1(x)
            _,x = self.conv3(x)
            x = self.en3(x)
            Com.append(_)
            en_list.append(x)
            x = self.pad1(x)
            _,x = self.conv4(x)
            x = self.en4(x)
            Com.append(_)
            en_list.append(x)
            x = self.pad1(x)
            _,x = self.conv5(x)
            x = self.en5(x)
            Com.append(_)
            en_list.append(x)
            return x, en_list,Com
        return x, en_list



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(1)
        self.chomp_t = Chomp_T(1)
        self.de5 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de4 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de3 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de2 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de1 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=1, kernel_size=(2, 5), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(1),
            nn.PReLU()
        )

    def forward(self, x, x_list):
        de_list = []
        x = self.de5(torch.cat((x, x_list[-1]), dim=1))
        de_list.append(x)
        x = self.de4(torch.cat((x, x_list[-2]), dim=1))
        de_list.append(x)
        x = self.de3(torch.cat((x, x_list[-3]), dim=1))
        de_list.append(x)
        x = self.de2(torch.cat((x, x_list[-4]), dim=1))
        de_list.append(x)
        x = self.de1(torch.cat((x, x_list[-5]), dim=1))
        de_list.append(x)
        return x, de_list


class Residual(nn.Module):
    def __init__(self,dilation):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=1)

        self.mainbranch = nn.Sequential(nn.PReLU(),
                                        nn.BatchNorm1d(64),
                                        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2 * dilation, dilation=dilation)
        )
        self.maskbranch = nn.Sequential(nn.PReLU(),
                                        nn.BatchNorm1d(64),
                                        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2 * dilation, dilation=dilation),
                                        nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.PReLU(),
                                   nn.BatchNorm1d(64),
                                   nn.Conv1d(in_channels=64, out_channels=256, stride=1, kernel_size=1))
    def forward(self,x):
        t = x
        x = self.conv1(x)
        x = self.mainbranch(x) * self.maskbranch(x)
        x = self.conv2(x)
        out = x + t
        return out


class TCM(nn.Module):
    def __init__(self):
        super(TCM, self).__init__()
        self.residual1 = Residual(dilation=1)
        self.residual2 = Residual(dilation=2)
        self.residual3 = Residual(dilation=4)
        self.residual4 = Residual(dilation=8)
        self.residual5 = Residual(dilation=16)
        self.residual6 = Residual(dilation=32)
    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.residual6(x)
        return x


class ConvGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvGLU, self).__init__()
        self.l = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.r = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.Sigmoid = nn.Sigmoid()
    def forward(self, inp, controller):
        left = self.l(inp)
        right = self.Sigmoid(self.r(controller))
        return left * right


class ConvTransGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTransGLU, self).__init__()
        self.l = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.r = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inp):
        left = self.l(inp)
        right = self.Sigmoid(self.r(inp))
        return left * right

class up_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(up_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, self.chomp_f:]


class down_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(down_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, :-self.chomp_f]


class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t

    def forward(self, x):
        return x[:, :, :-self.chomp_t, :]

class BiConvGLU(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size ,stride):
        super(BiConvGLU, self).__init__()
        self.con1 = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=1,stride=1)
        self.l = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=kernel_size,stride=stride)
        self.l_conv = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1)
        self.r = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=kernel_size,stride=stride)
        self.r_conv = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1)
        self.Sigmoid = nn.Sigmoid()
        self.con2 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self,inp,controller=None):
        inp = self.con1(inp)
        if controller != None:
            left = self.l(inp)
            right = self.r(inp)
            left_mask = self.Sigmoid(self.l_conv((left + controller)/2))
            right_mask = self.Sigmoid(self.r_conv((right + controller)/2))
            left = left * right_mask
            right = right * left_mask
            return self.con2(left + right)
        else:
            left = self.l(inp)
            right = self.r(inp)
            left_mask = self.Sigmoid(self.l_conv(left))
            right_mask = self.Sigmoid(self.r_conv(right))
            left = left * right_mask
            right = right * left_mask
        return left+right, self.con2(left + right)

class BiConvTransGLU(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size ,stride):
        super(BiConvTransGLU, self).__init__()
        self.con1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1)
        self.l = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=kernel_size,stride=stride)
        self.l_conv = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=1,stride=1)
        self.r_conv = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=1,stride=1)
        self.r = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=kernel_size,stride=stride)
        self.Sigmoid = nn.Sigmoid()
        self.con2 =nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self,inp):
        inp = self.con1(inp)
        left = self.l(inp)
        right = self.r(inp)
        left_mask = self.Sigmoid(self.l_conv(left))
        right_mask = self.Sigmoid(self.r_conv(right))
        left = left * right_mask
        right = right * left_mask
        return self.con2(left + right)