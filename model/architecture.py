"""
ForestSense — Siamese Multimodal Fusion Network
=================================================
Architecture:
  1. Shared ResNet-34 encoder processes T1 and T2 independently
  2. At each scale, T1/T2 features are concatenated + channel-attention fused
  3. UNet-style decoder with skip connections from fused features
  4. 4-class pixel-wise change map output

Input:
  t1: (B, 12, 256, 256)  — T1 SAR+Optical stack
  t2: (B, 12, 256, 256)  — T2 SAR+Optical stack

Output:
  logits: (B, 4, 256, 256)  — raw class logits (apply softmax for probabilities)

Classes:
  0 = Unchanged Non-Forest
  1 = Stable Forest
  2 = Forest Loss
  3 = Forest Gain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Building Blocks ────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Learns which fused channels to emphasise.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FusionBlock(nn.Module):
    """
    Fuse T1 and T2 features at one encoder scale.
    Strategy: cat([f_t1, f_t2, |f_t1 - f_t2|]) → Conv → ChannelAttention
    """
    def __init__(self, enc_ch: int):
        super().__init__()
        self.conv = ConvBnRelu(enc_ch * 3, enc_ch)
        self.attn = ChannelAttention(enc_ch)

    def forward(self, f_t1: torch.Tensor, f_t2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f_t1 - f_t2)
        x = torch.cat([f_t1, f_t2, diff], dim=1)   # (B, 3*C, H, W)
        x = self.conv(x)                             # (B, C, H, W)
        x = self.attn(x)
        return x


class DecoderBlock(nn.Module):
    """
    Upsample → Concatenate skip → ConvBnRelu × 2
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Main Model ─────────────────────────────────────────────────────────────────

class SiameseFusionNet(nn.Module):
    """
    Siamese Dual-Encoder Change Detection Network.

    ResNet-34 backbone gives 5 feature maps at strides 2,4,8,16,32.
    Channels at each stage: [64, 64, 128, 256, 512]

    For ResNet-50: [64, 256, 512, 1024, 2048]  (set backbone='resnet50')
    """

    BACKBONE_CH = {
        'resnet18' : [64, 64,  128, 256, 512],
        'resnet34' : [64, 64,  128, 256, 512],
        'resnet50' : [64, 256, 512, 1024, 2048],
    }

    def __init__(
        self,
        in_channels : int  = 12,
        num_classes : int  = 4,
        backbone    : str  = 'resnet18',   # resnet18 for ~2k patches; resnet34/50 for larger datasets
        pretrained  : bool = True,
        dropout     : float = 0.3,         # Higher dropout prevents overfitting on small datasets
    ):
        super().__init__()
        assert backbone in self.BACKBONE_CH, \
            f"backbone must be one of {list(self.BACKBONE_CH.keys())}"

        self.enc_ch = self.BACKBONE_CH[backbone]

        # ── Shared Encoder ─────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            backbone,
            pretrained    = pretrained,
            features_only = True,
            out_indices   = (0, 1, 2, 3, 4),
        )

        # Patch the first conv to accept in_channels bands
        old_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride      = old_conv.stride,
            padding     = old_conv.padding,
            bias        = False,
        )
        if pretrained:
            with torch.no_grad():
                # Average RGB weights → replicate across all in_channels
                avg_w = old_conv.weight.data.mean(dim=1, keepdim=True)
                self.encoder.conv1.weight.data = avg_w.repeat(1, in_channels, 1, 1)

        # ── Fusion Modules (one per encoder stage) ─────────────────────────────
        self.fuse = nn.ModuleList([FusionBlock(c) for c in self.enc_ch])

        # ── Decoder ────────────────────────────────────────────────────────────
        # Input to decoder is the deepest fused feature; skips come from shallower fused features
        ec = self.enc_ch   # shorthand
        self.dec4 = DecoderBlock(ec[4], ec[3], 256)
        self.dec3 = DecoderBlock(256,   ec[2], 128)
        self.dec2 = DecoderBlock(128,   ec[1], 64)
        self.dec1 = DecoderBlock(64,    ec[0], 32)

        # Final upsampling back to full resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ConvBnRelu(32, 32),
        )

        self.dropout   = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Kaiming init for decoder layers."""
        for m in [self.dec4, self.dec3, self.dec2, self.dec1,
                  self.final_up, self.classifier, self.fuse]:
            for layer in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def encode(self, x: torch.Tensor):
        """Extract 5 feature maps from the shared encoder."""
        return self.encoder(x)   # List of 5 tensors

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t1: (B, C, H, W) — T1 image stack
            t2: (B, C, H, W) — T2 image stack
        Returns:
            logits: (B, num_classes, H, W)
        """
        # Encode both time periods with shared weights
        f1 = self.encode(t1)   # [f1_0, f1_1, f1_2, f1_3, f1_4]
        f2 = self.encode(t2)   # [f2_0, f2_1, f2_2, f2_3, f2_4]

        # Fuse at each scale
        fused = [self.fuse[i](f1[i], f2[i]) for i in range(5)]
        # fused[0]: (B,  64, H/2,  W/2)
        # fused[1]: (B,  64, H/4,  W/4)
        # fused[2]: (B, 128, H/8,  W/8)
        # fused[3]: (B, 256, H/16, W/16)
        # fused[4]: (B, 512, H/32, W/32)

        # Decode (bottom-up)
        x = self.dec4(fused[4], fused[3])   # (B, 256, H/16, W/16)
        x = self.dec3(x,        fused[2])   # (B, 128, H/8,  W/8)
        x = self.dec2(x,        fused[1])   # (B,  64, H/4,  W/4)
        x = self.dec1(x,        fused[0])   # (B,  32, H/2,  W/2)

        x = self.dropout(x)
        x = self.final_up(x)                # (B, 32, H, W)
        logits = self.classifier(x)         # (B, num_classes, H, W)

        return logits


# ── Model Factory & Parameter Count ───────────────────────────────────────────

def build_model(
    in_channels : int   = 12,
    num_classes : int   = 4,
    backbone    : str   = 'resnet18',   # Optimal for ~2k patches
    pretrained  : bool  = True,
) -> SiameseFusionNet:
    """Build and return the model. Prints parameter count."""
    model = SiameseFusionNet(
        in_channels = in_channels,
        num_classes = num_classes,
        backbone    = backbone,
        pretrained  = pretrained,
    )
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ SiameseFusionNet ({backbone}) built.")
    print(f"   Total params    : {total:,}")
    print(f"   Trainable params: {trainable:,}")
    return model


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    t1 = torch.randn(2, 12, 256, 256).to(device)
    t2 = torch.randn(2, 12, 256, 256).to(device)
    with torch.no_grad():
        out = model(t1, t2)
    print(f"   Input  : {tuple(t1.shape)} × 2")
    print(f"   Output : {tuple(out.shape)}")
    assert out.shape == (2, 4, 256, 256), "Shape mismatch!"
    print("✅ Forward pass OK.")
