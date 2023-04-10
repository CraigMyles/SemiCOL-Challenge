import torch
from torch import nn, einsum
from vit_pytorch.crossformer import CrossFormer
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

# helpers

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)
    
# Override CrossFormerSegmentation
class CrossFormerSegmentation(nn.Module):
    def __init__(
        self,
        dim=(64, 128, 256, 512),
        *args,
        num_classes,
        upsample_mode,
        transpose_conv_out_channels=None,
        transpose_conv_kernel_size=None,
        transpose_conv_stride=None,
        **kwargs
    ):
        super().__init__()

        self.dim = cast_tuple(dim, 4)

        self.cross_former = CrossFormer(dim=self.dim, *args, num_classes=num_classes, **kwargs)
        self.segmentation_head = nn.Conv2d(self.dim[-1], num_classes, kernel_size=1)

        if upsample_mode == "bilinear" or upsample_mode == "nearest":
            self.upsample = lambda x, input_size: F.interpolate(x, size=input_size, mode=upsample_mode)
        elif upsample_mode == "transpose_conv":
            assert (
                transpose_conv_out_channels is not None
                and transpose_conv_kernel_size is not None
                and transpose_conv_stride is not None
            ), "Parameters for transpose convolution must be specified."
            self.upsample = nn.ConvTranspose2d(
                self.dim[-1],
                transpose_conv_out_channels,
                kernel_size=transpose_conv_kernel_size,
                stride=transpose_conv_stride,
            )
        else:
            raise ValueError("Invalid upsample mode specified.")


    def forward(self, x):
        input_size = x.shape[-2:]

        features = self.cross_former(x)

        # Print the output tensor from the CrossFormer model
        print("Features tensor:", features)

        # Print the shape of features tensor
        print("Features shape:", features.shape)

        # Calculate the output height and width
        output_height, output_width = x.shape[-2] // 4, x.shape[-1] // 4

        # Reshape the features tensor
        features = features.view(-1, self.dim[-1], output_height, output_width)

        logits = self.segmentation_head(features)
        segmentation_map = self.upsample(logits, input_size)
        return segmentation_map
