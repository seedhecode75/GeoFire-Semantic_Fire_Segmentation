class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class AttentionGate(nn.Module):
    def __init__(self, C_g, C_x, C_int):
        super().__init__()
        self.W_g, self.W_x = nn.Conv2d(C_g, C_int, 1, bias=False), nn.Conv2d(C_x, C_int, 1, bias=False)
        self.psi, self.bn = nn.Conv2d(C_int, 1, 1, bias=False), nn.BatchNorm2d(1)
    def forward(self, g, x):
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        attn = torch.sigmoid(self.bn(self.psi(F.relu(self.W_g(g) + self.W_x(x)))))
        return x * attn

class LowResEncoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.AdaptiveAvgPool2d((32, 32)), DoubleConv(in_channels, 32), nn.MaxPool2d(2), DoubleConv(32, out_channels), nn.MaxPool2d(2), nn.Conv2d(out_channels, out_channels, 3, padding=1))
    def forward(self, x): return self.encoder(x)

class MultiScaleFusionUNet(nn.Module):
    def __init__(self, in_high=3, in_low=2, out_channels=1):
        super().__init__()
        self.inc_high = DoubleConv(in_high, 64)
        self.down1, self.down2, self.down3, self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128)), nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)), nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512)), nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.low_enc = LowResEncoder(in_channels=in_low, out_channels=64)
        self.fuse5, self.fuse4, self.fuse3, self.fuse2 = nn.Conv2d(1088, 1024, 1), nn.Conv2d(576, 512, 1), nn.Conv2d(320, 256, 1), nn.Conv2d(192, 128, 1)
        self.up1, self.up2, self.up3, self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2), nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.cu1, self.cu2, self.cu3, self.cu4 = DoubleConv(1024, 512), DoubleConv(512, 256), DoubleConv(256, 128), DoubleConv(128, 64)
        self.ag1, self.ag2, self.ag3, self.ag4 = AttentionGate(512, 512, 256), AttentionGate(256, 256, 128), AttentionGate(128, 128, 64), AttentionGate(64, 64, 32)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x_high, x_low):
        x1 = self.inc_high(x_high); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        lf = self.low_enc(x_low)
        x5f = self.fuse5(torch.cat([x5, F.interpolate(lf, size=x5.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        x4f = self.fuse4(torch.cat([x4, F.interpolate(lf, size=x4.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        x3f = self.fuse3(torch.cat([x3, F.interpolate(lf, size=x3.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        x2f = self.fuse2(torch.cat([x2, F.interpolate(lf, size=x2.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        x = self.cu1(torch.cat([self.up1(x5f), self.ag1(self.up1(x5f), x4f)], dim=1))
        x = self.cu2(torch.cat([self.up2(x), self.ag2(self.up2(x), x3f)], dim=1))
        x = self.cu3(torch.cat([self.up3(x), self.ag3(self.up3(x), x2f)], dim=1))
        x = self.cu4(torch.cat([self.up4(x), self.ag4(self.up4(x), x1)], dim=1))
        return self.outc(x)
