import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadBlock(nn.Module):
    def __init__(self, in_channels = 1, mid_channels = None, out_channels = 64):
        super(HeadBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.GroupNorm(1, out_channels)
        )
    def forward(self, x):
        return self.double_conv(x)
    
class DownBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),  # Downsampling
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class TTS_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False, use_shortcut=True):
        super(TTS_block, self).__init__()
        self.use_attention = use_attention
        self.use_shortcut = use_shortcut
        self.relu = nn.ReLU()
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=1, batch_first=True)
        
        # Embedding for time (t) and label (y)
        self.embed_t = nn.Linear(1, out_channels)
        self.embed_y = nn.Linear(1, out_channels)

        # Normalization (helps with stability)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, t, y, x_down=None):
        
        t= t.to(torch.float32)

        
        
        x = torch.concat([x, x_down], dim=1) if x_down is not None else x  # Concatenate x and x_down for skip connection
        shortcut = self.conv1(x) if self.use_shortcut else None  # Save shortcut connection\
        
        x = self.conv1(x)
        # Main conv operations
        x = self.conv2(x)
        x = self.norm1(x)
        x = nn.ReLU()(x)

        # Embeddings for t and y
        t = t.reshape(-1, 1)
        y = y.reshape(-1, 1)
        t = self.embed_t(t).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        y = self.embed_y(y).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        

        x = x + t + y  # Add conditioning

        x = self.conv3(x)
        x = self.norm2(x)
        x = nn.ReLU()(x)

        # Apply shortcut connection
        if self.use_shortcut:
            x = x + shortcut

        # Apply attention (if enabled)
        if self.use_attention:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).permute(0, 2, 1)  # Reshape for MultiheadAttention
            x, _ = self.attention(x, x, x)
            x = x.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to 4D
        
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU()
        )
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class TailBlock(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(TailBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    


class UNet(nn.Module):

    def __init__(self, c_in=  1, c_out = 1, time_dim = 256, device = 'cpu'):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.device = device
        #Head
        self.head = HeadBlock(in_channels=c_in, out_channels=64)

        #Downsampling
        self.tts_d1 = TTS_block(in_channels=64, out_channels=64, use_attention=True, use_shortcut=True)
        self.tts_d2 = TTS_block(in_channels=64, out_channels=64, use_attention=True, use_shortcut=True)
        self.down1 = DownBlock(in_channels=64, out_channels=64)
        self.tts_d3 = TTS_block(in_channels=64, out_channels=128, use_attention=True, use_shortcut=False)
        self.tts_d4 = TTS_block(in_channels=128, out_channels=128, use_attention=True, use_shortcut=True)
        self.down2 = DownBlock(in_channels=128, out_channels=128)
        self.tts_d5 = TTS_block(in_channels=128, out_channels=256, use_attention=True, use_shortcut=True)
        self.tts_d6 = TTS_block(in_channels=256, out_channels=256, use_attention=True, use_shortcut=True)

        #Bottleneck
        self.tts_bottleneck1 = TTS_block(in_channels=256, out_channels=256, use_attention=True, use_shortcut=True)
        self.tts_bottleneck2 = TTS_block(in_channels=256, out_channels=256, use_attention=False, use_shortcut=True)

        #Upsampling
        self.tts_u1 = TTS_block(in_channels=512, out_channels=256, use_attention=False, use_shortcut=True)
        self.tts_u2 = TTS_block(in_channels=512, out_channels=256, use_attention=False, use_shortcut=True)
        self.tts_u3 = TTS_block(in_channels=384, out_channels=256, use_attention=False, use_shortcut=True)
        self.up1 = UpBlock(in_channels=256, out_channels=256)
        self.tts_u4 = TTS_block(in_channels=384, out_channels=128, use_attention=False, use_shortcut=True)
        self.tts_u5 = TTS_block(in_channels=256, out_channels=128, use_attention=False, use_shortcut=True)
        self.tts_u6 = TTS_block(in_channels=192, out_channels=128, use_attention=False, use_shortcut=True)
        self.up2 = UpBlock(in_channels=128, out_channels=128)
        self.tts_u7 = TTS_block(in_channels=192, out_channels=64, use_attention=False, use_shortcut=True)
        self.tts_u8 = TTS_block(in_channels=128, out_channels=64, use_attention=False, use_shortcut=True)
        self.tts_u9 = TTS_block(in_channels=128, out_channels=64, use_attention=False, use_shortcut=True)

        #Tail
        self.tail = TailBlock(in_channels=64, out_channels=c_out)
    

    #dont need this due to tts blocks I think?
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device.float() / channels))
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t, l):
        x1 = self.head(x)
        x2 = self.tts_d1(x1, t, l)
        x3 = self.tts_d2(x2, t, l)
        x4 = self.down1(x3)
        x5 = self.tts_d3(x4, t, l)
        x6 = self.tts_d4(x5, t, l)
        x7 = self.down2(x6)
        x8 = self.tts_d5(x7, t, l)
        x9 = self.tts_d6(x8, t, l)
        x10 = self.tts_bottleneck1(x9, t, l)
        x11 = self.tts_bottleneck2(x10, t, l)
        x12 = self.tts_u1(x11, t, l, x9)
        x13 = self.tts_u2(x12, t, l, x8)
        x14 = self.tts_u3(x13, t, l, x7)
        x15 = self.up1(x14)
        x16 = self.tts_u4(x15, t, l, x6)
        x17 = self.tts_u5(x16, t, l, x5)
        x18 = self.tts_u6(x17, t, l, x4)
        x19 = self.up2(x18)
        x20 = self.tts_u7(x19, t, l, x3)
        x21 = self.tts_u8(x20, t, l, x2)
        x22 = self.tts_u9(x21, t, l, x1)

        x23 = self.tail(x22)

        return x23







        




        




        
