import math
import torch
from torch import Tensor
from torch import nn
import torch.utils
import torch.utils.checkpoint
import warnings
from itertools import repeat
from typing import Tuple


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        ll = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * ll - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class PromptGenerator(nn.Module):
    def __init__(
        self,
        scale_factor=32,
        embed_dim=1280,
        depth=32,
        freq_nums=0.25,
        img_size=1024,
        patch_size=16,
    ):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.freq_nums = freq_nums
        self.depth = depth

        # Implementation for MLP up
        self.shared_mlp = nn.Linear(
            self.embed_dim // self.scale_factor, self.embed_dim
        )
        self.embedding_generator = nn.Linear(
            self.embed_dim, self.embed_dim // self.scale_factor
        )
        # Implementation for MLP tune
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(
                    self.embed_dim // self.scale_factor,
                    self.embed_dim // self.scale_factor,
                ),
                nn.GELU(),
            )
            setattr(self, "lightweight_mlp_{}".format(str(i)), lightweight_mlp)

        self.prompt_generator = PatchEmbed2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=self.embed_dim // self.scale_factor,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape
        x = x.reshape(N, C, H * W).permute(0, 2, 1)
        return self.embedding_generator(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums)
        return self.prompt_generator(x)

    def get_prompt(self, handcrafted_feature, embedding_feature):
        N, C, H, W = handcrafted_feature.shape
        handcrafted_feature = handcrafted_feature.view(N, C, H * W).permute(
            0, 2, 1
        )
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(
                self, "lightweight_mlp_{}".format(str(i))
            )
            # prompt = proj_prompt(prompt)
            prompt = lightweight_mlp(handcrafted_feature + embedding_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts

    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** 0.5 // 2)
        mask[
            :, :, w // 2 - line : w // 2 + line, h // 2 - line : h // 2 + line
        ] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        fft = fft * (1 - mask)
        # fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv


def to_2tuple(x) -> Tuple:
    if isinstance(x, Tuple):
        return x
    return tuple(repeat(x, 2))


class PatchEmbed2(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=1024, patch_size=16, in_chans=3, embed_dim=1280
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x
