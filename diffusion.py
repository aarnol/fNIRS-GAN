import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadBlock(nn.Module):
    def __init__(self, in_channels=1, mid_channels=None, out_channels=64):
        super(HeadBlock, self).__init__()
        if mid_channels is None:
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
    def __init__(self, in_channels, out_channels, num_classes, use_attention=False, use_shortcut=True):
        super(TTS_block, self).__init__()
        self.use_attention = use_attention
        self.use_shortcut = use_shortcut

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, batch_first=True)

        # Embedding for time (t) and label (y)
        self.embed_t = nn.Linear(1, out_channels)  # Time embedding
        self.embed_y = nn.Linear(num_classes, out_channels)  # One-hot label embedding

        # Normalization
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        # Activation function
        self.activation = nn.GELU()

    def forward(self, x, t, y_onehot, x_down=None):
        """
        x: Input feature map [B, C, H, W]
        t: Time step (scalar) [B, 1]
        y_onehot: One-hot encoded class labels [B, num_classes]
        x_down: Optional skip connection input
        """
        if x_down is not None:
            x = torch.cat([x, x_down], dim=1)

        # Shortcut connection
        shortcut = self.conv1(x) if self.use_shortcut else None

        # Main forward path
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Conditioning with time and one-hot label embeddings
        t = t.to(torch.float32).reshape(-1, 1)
        t_emb = self.embed_t(t).unsqueeze(-1).unsqueeze(-1).expand_as(x)

        y_emb = self.embed_y(y_onehot).unsqueeze(-1).unsqueeze(-1).expand_as(x)

        x = x + t_emb + y_emb  # Add conditioning

        x = self.conv3(x)
        x = self.norm2(x)
        x = self.activation(x)

        # Apply shortcut connection
        if self.use_shortcut:
            x = x + shortcut

        # Apply attention if enabled
        if self.use_attention:
            B, C, H, W = x.shape
            x_flat = x.flatten(2).permute(0, 2, 1)  # Reshape for MultiheadAttention: [B, HW, C]
            x_attn, _ = self.attention(x_flat, x_flat, x_flat)
            x = x_attn.permute(0, 2, 1).view(B, C, H, W)  # Reshape back to [B, C, H, W]

        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class TailBlock(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(TailBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, in_channels // 2), in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device='cpu'):
        super(UNet, self).__init__()
        self.device = device
        
        # Initial feature dimensions
        init_features = 64
        
        # Head - Initial processing
        self.head = HeadBlock(in_channels=c_in, out_channels=init_features)

        # Encoder (downsampling path)
        # First block
        self.tts_d1 = TTS_block(in_channels=init_features, out_channels=init_features, use_attention=True)
        self.tts_d2 = TTS_block(in_channels=init_features, out_channels=init_features, use_attention=True)
        self.down1 = DownBlock(in_channels=init_features, out_channels=init_features)
        
        # Second block
        self.tts_d3 = TTS_block(in_channels=init_features, out_channels=init_features*2, use_attention=True, use_shortcut=False)
        self.tts_d4 = TTS_block(in_channels=init_features*2, out_channels=init_features*2, use_attention=True)
        self.down2 = DownBlock(in_channels=init_features*2, out_channels=init_features*2)
        
        # Third block
        self.tts_d5 = TTS_block(in_channels=init_features*2, out_channels=init_features*4, use_attention=True)
        self.tts_d6 = TTS_block(in_channels=init_features*4, out_channels=init_features*4, use_attention=True)

        # Bottleneck
        self.tts_bottleneck1 = TTS_block(in_channels=init_features*4, out_channels=init_features*4, use_attention=True)
        self.tts_bottleneck2 = TTS_block(in_channels=init_features*4, out_channels=init_features*4, use_attention=False)

        # Decoder (upsampling path)
        # First block (deepest)
        self.tts_u1 = TTS_block(in_channels=init_features*8, out_channels=init_features*4, use_attention=False)
        self.tts_u2 = TTS_block(in_channels=init_features*8, out_channels=init_features*4, use_attention=False)
        self.tts_u3 = TTS_block(in_channels=init_features*6, out_channels=init_features*4, use_attention=False)
        self.up1 = UpBlock(in_channels=init_features*4, out_channels=init_features*4)
        
        # Second block
        self.tts_u4 = TTS_block(in_channels=init_features*6, out_channels=init_features*2, use_attention=False)
        self.tts_u5 = TTS_block(in_channels=init_features*4, out_channels=init_features*2, use_attention=False)
        self.tts_u6 = TTS_block(in_channels=init_features*3, out_channels=init_features*2, use_attention=False)
        self.up2 = UpBlock(in_channels=init_features*2, out_channels=init_features*2)
        
        # Third block (shallowest)
        self.tts_u7 = TTS_block(in_channels=init_features*3, out_channels=init_features, use_attention=False)
        self.tts_u8 = TTS_block(in_channels=init_features*2, out_channels=init_features, use_attention=False)
        self.tts_u9 = TTS_block(in_channels=init_features*2, out_channels=init_features, use_attention=False)

        # Output layer
        self.tail = TailBlock(in_channels=init_features, out_channels=c_out)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, t, l):
        # Store skip connections for the decoder
        skip_connections = []
        
        # Encoder path
        x1 = self.head(x)
        skip_connections.append(x1)
        
        x2 = self.tts_d1(x1, t, l)
        skip_connections.append(x2)
        
        x3 = self.tts_d2(x2, t, l)
        skip_connections.append(x3)
        
        x4 = self.down1(x3)
        skip_connections.append(x4)
        
        x5 = self.tts_d3(x4, t, l)
        skip_connections.append(x5)
        
        x6 = self.tts_d4(x5, t, l)
        skip_connections.append(x6)
        
        x7 = self.down2(x6)
        skip_connections.append(x7)
        
        x8 = self.tts_d5(x7, t, l)
        skip_connections.append(x8)
        
        x9 = self.tts_d6(x8, t, l)
        skip_connections.append(x9)
        
        # Bottleneck
        x10 = self.tts_bottleneck1(x9, t, l)
        x11 = self.tts_bottleneck2(x10, t, l)
        
        # Decoder path with skip connections
        x12 = self.tts_u1(x11, t, l, skip_connections[-1])  # x9
        x13 = self.tts_u2(x12, t, l, skip_connections[-2])  # x8
        x14 = self.tts_u3(x13, t, l, skip_connections[-3])  # x7
        x15 = self.up1(x14)
        
        x16 = self.tts_u4(x15, t, l, skip_connections[-4])  # x6
        x17 = self.tts_u5(x16, t, l, skip_connections[-5])  # x5
        x18 = self.tts_u6(x17, t, l, skip_connections[-6])  # x4
        x19 = self.up2(x18)
        
        x20 = self.tts_u7(x19, t, l, skip_connections[-7])  # x3
        x21 = self.tts_u8(x20, t, l, skip_connections[-8])  # x2
        x22 = self.tts_u9(x21, t, l, skip_connections[-9])  # x1
        
        # Final output
        x23 = self.tail(x22)
        
        return x23