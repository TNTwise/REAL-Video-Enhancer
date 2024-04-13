import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import MyPixelShuffle, MyPReLU


# Residual Block
def ResidualBlock(in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        MyPReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        MyPReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    )


# downsample block
def DownsampleBlock(in_channels, out_channels, stride=2):
    return torch.nn.Sequential(
        MyPReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        MyPReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    )


# upsample block
def UpsampleBlock(in_channels, out_channels, stride=2):
    return torch.nn.Sequential(
        MyPReLU(),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True),
        MyPReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    )


class PixelShuffleBlcok(nn.Module):
    def __init__(self, in_feat, num_feat, num_out_ch):
        super(PixelShuffleBlcok, self).__init__()
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(in_feat, num_feat, 3, 1, 1),
            MyPReLU()
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
            MyPixelShuffle(2)
        )
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x

# grid network
class GridNet(nn.Module):
    def __init__(self, in_channels=12, in_channels1=128, in_channels2=256, in_channels3=384, out_channels=3):
        super(GridNet, self).__init__()

        self.residual_model_head = ResidualBlock(in_channels, 64)
        self.residual_model_head1 = ResidualBlock(in_channels1, 64)
        self.residual_model_head2 = ResidualBlock(in_channels2, 128)
        self.residual_model_head3 = ResidualBlock(in_channels3, 192)

        self.residual_model_01=ResidualBlock(64, 64)
        #self.residual_model_02=ResidualBlock(64, 64)
        #self.residual_model_03=ResidualBlock(64, 64)
        self.residual_model_04=ResidualBlock(64, 64)
        self.residual_model_05=ResidualBlock(64, 64)
        self.residual_model_tail=PixelShuffleBlcok(64, 64, out_channels)


        self.residual_model_11=ResidualBlock(128, 128)
        #self.residual_model_12=ResidualBlock(128, 128)
        #self.residual_model_13=ResidualBlock(128, 128)
        self.residual_model_14=ResidualBlock(128, 128)
        self.residual_model_15=ResidualBlock(128, 128)

        self.residual_model_21=ResidualBlock(192, 192)
        #self.residual_model_22=ResidualBlock(192, 192)
        #self.residual_model_23=ResidualBlock(192, 192)
        self.residual_model_24=ResidualBlock(192, 192)
        self.residual_model_25=ResidualBlock(192, 192)

        #

        self.downsample_model_10=DownsampleBlock(64, 128)
        self.downsample_model_20=DownsampleBlock(128, 192)

        self.downsample_model_11=DownsampleBlock(64, 128)
        self.downsample_model_21=DownsampleBlock(128, 192)

        #self.downsample_model_12=DownsampleBlock(64, 128)
        #self.downsample_model_22=DownsampleBlock(128, 192)

        #

        #self.upsample_model_03=UpsampleBlock(128, 64)
        #self.upsample_model_13=UpsampleBlock(192, 128)

        self.upsample_model_04=UpsampleBlock(128, 64)
        self.upsample_model_14=UpsampleBlock(192, 128)

        self.upsample_model_05=UpsampleBlock(128, 64)
        self.upsample_model_15=UpsampleBlock(192, 128)

    def forward(self, x, x1, x2, x3):
        X00=self.residual_model_head(x) + self.residual_model_head1(x1)      #---   182 ~ 185
        # X10 = self.residual_model_head1(x1)

        X01=self.residual_model_01(X00) + X00#---   208 ~ 211 ,AddBackward1213

        X10=self.downsample_model_10(X00) + self.residual_model_head2(x2)   #---   186 ~ 189
        X20=self.downsample_model_20(X10) + self.residual_model_head3(x3)  #---   190 ~ 193

        residual_11=self.residual_model_11(X10) + X10  #201 ~ 204    , sum  AddBackward1206
        downsample_11=self.downsample_model_11(X01)    #214 ~ 217
        X11=residual_11 + downsample_11      #---      AddBackward1218

        residual_21=self.residual_model_21(X20) + X20  #194 ~ 197  ,   sum  AddBackward1199
        downsample_21=self.downsample_model_21(X11)    #219 ~ 222
        X21=residual_21 + downsample_21                # AddBackward1223


        X24=self.residual_model_24(X21) + X21 #---   224 ~ 227 , AddBackward1229
        X25=self.residual_model_25(X24) + X24 #---   230 ~ 233 , AddBackward1235


        upsample_14=self.upsample_model_14(X24)       #242 ~ 246
        residual_14=self.residual_model_14(X11) + X11 #248 ~ 251, AddBackward1253
        X14=upsample_14 + residual_14   #---   AddBackward1254

        upsample_04=self.upsample_model_04(X14)       #268 ~ 272
        residual_04=self.residual_model_04(X01) + X01 #274 ~ 277, AddBackward1279
        X04=upsample_04 + residual_04   #---  AddBackward1280

        upsample_15=self.upsample_model_15(X25)       #236 ~ 240
        residual_15=self.residual_model_15(X14) + X14 #255 ~ 258, AddBackward1260
        X15=upsample_15 + residual_15   # AddBackward1261

        upsample_05=self.upsample_model_05(X15)   # 262 ~ 266
        residual_05=self.residual_model_05(X04) + X04  #281 ~ 284,AddBackward1286
        X05=upsample_05 + residual_05  # AddBackward1287

        X_tail=self.residual_model_tail(X05)    #288 ~ 291

        return X_tail
