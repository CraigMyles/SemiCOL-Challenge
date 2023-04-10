import torch
import torch.nn as nn
from vit_pytorch import SimpleViT
from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
def posemb_sincos_2d(patches, h, w, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

class SimpleViTSegmentation(SimpleViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=dim_head
        )
        
        self.patch_height, self.patch_width = pair(image_size)
        self.num_patches_h = (self.patch_height // patch_size)
        self.num_patches_w = (self.patch_width // patch_size)
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.conv_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.LayerNorm([1, dim, 1, 1]),
            nn.Conv2d(dim, num_classes, kernel_size=1)
        )
    
    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        # pe = posemb_sincos_2d()
        pe = posemb_sincos_2d(x, h=self.num_patches_h, w=self.num_patches_w)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        
        # Reshape the output to have the same spatial dimensions as the patches
        # x = rearrange(x, 'b (h w) d -> b d h w', h=self.patch_height, w=self.patch_width)
        x = rearrange(x, 'b (h w) d -> b d h w', h=self.num_patches_h, w=self.num_patches_w)


        # Apply the convolutional head
        x = self.conv_head(x)

        # Normalize x along the channel, height, and width dimensions
        x = torch.nn.functional.layer_norm(x, x.shape[1:])
        
        # Resize the output to have the same spatial dimensions as the input image
        x = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return x
