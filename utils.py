import random
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.nn.functional as F
from typing import Tuple, List


def mask_iou(mask1: np.ndarray,
             mask2: np.ndarray) -> List[float]:
    """Calculate IoU for two masks
    mask1, mask2: [B, H, W]
    Return: iou: [B]
    """
    bt_size = mask1.shape[0]
    assert bt_size == mask2.shape[0]
    assert mask1.shape == mask2.shape
    iou = []
    for i in range(bt_size):
        mask1_i = mask1[i].reshape(-1).astype(int)
        mask2_i = mask2[i].reshape(-1).astype(int)
        intersection = np.sum(mask1_i * mask2_i)
        union = np.sum(mask1_i) + np.sum(mask2_i) - intersection + 1e-5
        iou.append(intersection / union)
    return iou


def DSC(mask1, mask2):
    """Dice similarity coefficient
    mask1, mask2: [B, H, W]
    Return: dsc: [B]
    """
    bt_size = mask1.shape[0]
    assert bt_size == mask2.shape[0]
    assert mask1.shape == mask2.shape
    dsc = []
    for i in range(bt_size):
        mask1_i = mask1[i].reshape(-1).astype(int)
        mask2_i = mask2[i].reshape(-1).astype(int)
        intersection = np.sum(mask1_i * mask2_i)
        union = np.sum(mask1_i) + np.sum(mask2_i) + 1e-5
        dsc.append(2 * intersection / union)
    return dsc


def set_random_seed(seed=1):
    """Set random seed and cuda backen for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def postprocess_masks(masks: torch.Tensor,
                      input_size: Tuple[int, ...],
                      original_size: Tuple[int, ...], image_size=1024) -> torch.Tensor:
    """
    Remove padding and upscale masks to the original image size.

    Args:
      masks (torch.Tensor):
        Batched masks from the mask_decoder, in BxCxHxW format.
      input_size (tuple(int, int)):
        The size of the image input to the model, in (H', W') format. Used to remove padding.
      original_size (tuple(int, int)):
        The original size of the image before resizing for input to the model, in (H, W) format.

    Returns:
      (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(
        masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size,
                          mode="bilinear", align_corners=False)
    return masks
