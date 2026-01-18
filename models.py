from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )
    def forward(self, x):
        return self.encoder_block(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels,  in_channels//2, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(in_channels//2),
            activation,
            nn.Conv2d(in_channels//2, out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            activation
        )
    def forward(self, x):
        return self.decoder_block(x)

class UNet(nn.Module):
    def __init__(self, in_channels =3, out_channels =3, n_filters = 32, activation=nn.ReLU()):
        super().__init__()
        
        # Config
        self.in_channels  =in_channels   
        self.out_channels = out_channels   
        self.n_filters    = n_filters  # scaled down from 64 in original paper
        self.activation   = activation
        
        # Up and downsampling methods
        self.downsample  = nn.MaxPool2d((2,2), stride=2)
        self.upsample    = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # encoder
        self.enc_block_1 = EncoderBlock(in_channels, 1*n_filters, activation)
        self.enc_block_2 = EncoderBlock(1*n_filters, 2*n_filters, activation)
        self.enc_block_3 = EncoderBlock(2*n_filters, 4*n_filters, activation)
        self.enc_block_4 = EncoderBlock(4*n_filters, 8*n_filters, activation)
        
        #bottleneck
        self.bottleneck  = nn.Sequential(
            nn.Conv2d( 8*n_filters, 16*n_filters, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(16*n_filters),
            activation,
            nn.Dropout2d(p=0.5),
            
            nn.Conv2d(16*n_filters,  8*n_filters, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(8*n_filters),
            activation,
            nn.Dropout2d(p=0.5)
        )
        
        # Decoder
        self.dec_block_4 = DecoderBlock(16*n_filters, 4*n_filters, activation)
        self.dec_block_3 = DecoderBlock( 8*n_filters, 2*n_filters, activation)
        self.dec_block_2 = DecoderBlock( 4*n_filters, 1*n_filters, activation)
        self.dec_block_1 = DecoderBlock( 2*n_filters, 1*n_filters, activation)
        
        # output projection
        self.output = nn.Conv2d(1*n_filters,  out_channels, kernel_size=(1,1), stride=1, padding=0)

        
    def forward(self, x):
        # Encoder
        skip_1 = self.enc_block_1(x)
        x      = self.downsample(skip_1)
        skip_2 = self.enc_block_2(x)
        x      = self.downsample(skip_2)
        skip_3 = self.enc_block_3(x)
        x      = self.downsample(skip_3)
        skip_4 = self.enc_block_4(x)
        x      = self.downsample(skip_4)
        
        # Bottleneck
        x      = self.bottleneck(x)
        
        # Decoder
        x      = self.upsample(x)
        x      = torch.cat((x, skip_4), axis=1)  # Skip connection
        x      = self.dec_block_4(x)
        x      =self.upsample(x)
        x      =  torch.cat((x, skip_3), axis=1)  # Skip connection
        x      = self.dec_block_3(x)
        x      =self.upsample(x)
        x      = torch.cat((x, skip_2), axis=1)  # Skip connection
        x      =  self.dec_block_2(x)
        x      = self.upsample(x)
        x      =torch.cat((x, skip_1), axis=1)  # Skip connection
        x      = self.dec_block_1(x)
        # x = self.output(x)
        logits = self.output(x)
        return logits
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params:,}\n')

def save_model(model, path='model_weights.pth'):
    torch.save(model.state_dict(), path)


#Unet with a ResNet Encoder

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pretrained=True):
        super().__init__()
        self.out_channels = out_channels
        
        # Load pretrained ResNet34
        resnet = models.resnet34(pretrained=pretrained)
        
        # Encoder blocks from ResNet34
        self.enc_block_1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.enc_block_2 = nn.Sequential(resnet.layer1)                          # 64 channels
        self.enc_block_3 = nn.Sequential(resnet.layer2)                          # 128 channels
        self.enc_block_4 = nn.Sequential(resnet.layer3)                          # 256 channels
        self.enc_block_5 = nn.Sequential(resnet.layer4)                          # 512 channels
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        
        # Decoder blocks
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # ResNet34: 64->64->128->256->512
        self.dec_block_5 = self._decoder_block(512 + 512, 256)   # 1024 -> 256
        self.dec_block_4 = self._decoder_block(256 + 256, 256)   # 512 -> 256
        self.dec_block_3 = self._decoder_block(256 + 128, 128)   # 384 -> 128
        self.dec_block_2 = self._decoder_block(128 + 64, 64)     # 192 -> 64
        self.dec_block_1 = self._decoder_block(64 + 64, 32)      # 128 -> 32
        
        # Output projection
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder (ResNet34: 64->64->128->256->512)
        skip_1 = self.enc_block_1(x)         # 1/1, 64 canaux
        x = nn.functional.max_pool2d(skip_1, kernel_size=2, stride=2)
        
        skip_2 = self.enc_block_2(x)         # 1/2, 64 canaux
        x = nn.functional.max_pool2d(skip_2, kernel_size=2, stride=2)
        
        skip_3 = self.enc_block_3(x)         # 1/4, 128 canaux
        x = nn.functional.max_pool2d(skip_3, kernel_size=2, stride=2)
        
        skip_4 = self.enc_block_4(x)         # 1/8, 256 canaux
        x = nn.functional.max_pool2d(skip_4, kernel_size=2, stride=2)
        
        skip_5 = self.enc_block_5(x)         # 1/16, 512 canaux
        x = nn.functional.max_pool2d(skip_5, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)               # 1/32
        
        # Decoder (up: 512->256->256->128->64->32)
        x = self.upsample(x)                 # 1/16
        x = torch.cat([x, skip_5], dim=1)    # 512 + 512 = 1024
        x = self.dec_block_5(x)              # 1024 -> 256
        
        x = self.upsample(x)                 # 1/8
        x = torch.cat([x, skip_4], dim=1)    # 256 + 256 = 512
        x = self.dec_block_4(x)              # 512 -> 256
        
        x = self.upsample(x)                 # 1/4
        x = torch.cat([x, skip_3], dim=1)    # 256 + 128 = 384
        x = self.dec_block_3(x)              # 384 -> 128
        
        x = self.upsample(x)                 # 1/2
        x = torch.cat([x, skip_2], dim=1)    # 128 + 64 = 192
        x = self.dec_block_2(x)              # 192 -> 64
        
        x = self.upsample(x)                 # 1/1
        x = torch.cat([x, skip_1], dim=1)    # 64 + 64 = 128
        x = self.dec_block_1(x)              # 128 -> 32
        
        logits = self.output(x)
        return logits
    


# TransUNet 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                           attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransUNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=22, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.1,
                 image_size=320):
        super().__init__()
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.patch_size = 16
        patches_size = (image_size // 4 // self.patch_size) ** 2
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, patches_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=drop_rate
            )
            for _ in range(depth)
        ])
        
        self.decoder_channels = (256, 128, 64, 32)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim if i == 0 else self.decoder_channels[i-1],
                                   self.decoder_channels[i],
                                   kernel_size=2,
                                   stride=2),
                nn.BatchNorm2d(self.decoder_channels[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.decoder_channels[i], self.decoder_channels[i],
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(self.decoder_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(4)
        ])
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        for decoder_block in self.decoder:
            x = decoder_block(x)
            
        x = self.final_conv(x)
        x = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=True)
        
        return x
    

#DenseUNet

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        
    def forward(self, x):
        return torch.cat([x, self.layers(x)], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.layers(x)

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.convTrans(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DenseUNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=22, growth_rate=32, 
                 block_config=(4, 4, 4, 4, 4), init_features=64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        channels = init_features
        self.encoder_blocks = nn.ModuleList()
        self.skip_connections_channels = []
        self.transition_downs = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(channels, growth_rate, num_layers)
            self.encoder_blocks.append(block)
            channels += num_layers * growth_rate
            self.skip_connections_channels.append(channels)
            
            if i != len(block_config) - 1:
                td = TransitionDown(channels, channels // 2)
                self.transition_downs.append(td)
                channels = channels // 2

        self.decoder_blocks = nn.ModuleList()
        self.transition_ups = nn.ModuleList()
        
        for i in range(len(block_config)-2, -1, -1):
            tu = TransitionUp(channels, self.skip_connections_channels[i])
            self.transition_ups.append(tu)
            
            decoder_channels = 2 * self.skip_connections_channels[i]
            block = DenseBlock(decoder_channels, growth_rate, block_config[i])
            self.decoder_blocks.append(block)
            channels = decoder_channels + block_config[i] * growth_rate
        
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_classes, kernel_size=1, bias=False)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        encoder_features = []
        
        td_index = 0
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_features.append(x)
            if i < len(self.transition_downs):
                x = self.transition_downs[i](x)
        
        for i in range(len(self.decoder_blocks)):
            skip_feat = encoder_features[-(i+2)]
            
            x = self.transition_ups[i](x)
            x = torch.cat([x, skip_feat], dim=1)
            x = self.decoder_blocks[i](x)

        x = self.final_conv(x)
        
        if x.size(-1) != 320 or x.size(-2) != 320:
            x = F.interpolate(x, size=(320, 320), mode='bilinear', align_corners=True)
        return x