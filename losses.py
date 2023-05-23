import torch
import torch.backends
import torch.backends.cudnn
from torch import Tensor
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers.models.maskformer.modeling_maskformer \
    import dice_loss, sigmoid_focal_loss
from monai.losses.dice import DiceCELoss
from utils import postprocess_masks
from typing import Callable


def unfold_wo_center(x: Tensor,
                     kernel_size: int,
                     dilation: int) -> Tensor:
    '''Unfold an input tensor into a matrix, while removing the center pixel of
        each patch that corresponds to a convolutional kernel.
    Args:
        x: [bt_size, c, h, w]
        kernel_size: Size of the convolutional kernel
        dilation: Dilation rate to use

    Returns:
        unfolded_x: [bt_size, c, kernel_size^2 - 1, h, w]
    '''
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # Compute the amount of padding required for "SAME" padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    # unfolded_x: [bt_size, c * kernel_size^2, h * w]
    unfolded_x: Tensor = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    # unfolded_x: [bt_size, c, kernel_size^2, h, w]
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels,
    # ex. if the kernel size = 3, the resulting sequence will be a \
    #     square with dimensions 3x3 - 1, and the center pixel will be removed.
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)
    return unfolded_x


def get_images_color_similarity(
        images: Tensor,
        kernel_size: int = 3,
        dilation: int = 1) -> Tensor:
    '''
    Args:
        images: [bt_size, c, h, w]
        kernel_size: Size of the convolutional kernel
        dilation: Dilation rate to use
    Returns:
        similarity: [bt_size, kernel_size^2 - 1, h ,w]
    '''
    assert images.dim() == 4

    unfolded_images: Tensor = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    # Compute the distance between a pixel and its 8 adjacent pixels.
    diff: Tensor = images[:, :, None] - unfolded_images
    similarity: Tensor = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    # unfolded_weights = unfold_wo_center(
    #     image_masks[None, None], kernel_size=kernel_size,
    #     dilation=dilation
    # )
    # unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity


def compute_pairwise_term(mask_logits: Tensor,
                          pairwise_size: int = 3,
                          pairwise_dilation: int = 1) -> Tensor:
    ''' The pairwise term is used to enforce consistency between predictions
    for adjacent pixels in an image.

    Args:
        mask_logits: [bt_size, nb_class, h, w]
        pairwise_size: the size of the converlutional kernel
        pairwise_dilation: the size of the dilation rate
    Returns:
        log_same_prob: [bt_size, kernel_size^2 - 1, h, w]
    '''
    assert mask_logits.dim() == 4

    # prob of being the foreground
    log_fg_prob: Tensor = F.logsigmoid(mask_logits)
    # prob of being the background
    log_bg_prob: Tensor = F.logsigmoid(-mask_logits)

    # pairwise
    log_fg_prob_unfold: Tensor = unfold_wo_center(
        log_fg_prob,
        kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold: Tensor = unfold_wo_center(
        log_bg_prob,
        kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction \
    #   = p_i * p_j + (1 - p_i) * (1 - p_j)
    # compute the the probability in log space to avoid numerical instability
    log_same_fg_prob: Tensor = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob: Tensor = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_: Tensor = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob: Tensor = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def pairwise_loss(mask_logits: Tensor,
                  gt_mask: Tensor,
                  images: Tensor,
                  pairwise_size: int = 3,
                  pairwise_dilation: int = 1) -> Tensor:
    '''
    Args:
        mask_logit: [bt_size, nb_class, h, w]
        gt_mask: [bt_size, nb_class, h, w]
        images: cv2.Mat, LAB format
        pairwise_size: the size of the converlutional kernel
        pairwise_dilation: the size of the dilation rate

    Returns:
        loss_pairwise: Tensor
    '''
    gt_mask = gt_mask.detach().clone()
    images_color_similarity = get_images_color_similarity(images)
    pairwise_prob: Tensor = compute_pairwise_term(
        mask_logits,
        pairwise_size,
        pairwise_dilation)
    '''
    TODO:
    0.3 is the threshold,
    can i use mean(image_color_similarity) instead?
    '''
    weights: Tensor = (images_color_similarity
                       * gt_mask.float() > 0.3).float()
    # weights: Tensor = (images_color_similarity >= 0.3).\
    #     float() * gt_mask.float()
    loss_pairwise: Tensor = (pairwise_prob * weights).sum() \
        / weights.sum().clamp(min=1.0)

    return loss_pairwise


def critn_pair(outputs, gt_mask, batch,
               pairwise_size: int = 3,
               pairwise_dilation: int = 1) -> Tensor:
    low_res_masks = outputs.pred_masks
    upscaled_masks = postprocess_masks(
        low_res_masks.squeeze(1),
        batch["reshaped_input_sizes"][0].tolist(),
        batch["original_sizes"][0].tolist()).to(gt_mask.device)
    'process upscaled masks'
    '''Compute iou by thresholding
    predicted_masks = threshold(upscaled_masks, 0.0, 0)
    predicted_masks = normalize(
                threshold(upscaled_masks, 0.0, 0))
    '''
    gt_mask = F.normalize(F.threshold(upscaled_masks, 0.0, 0)).float()
    mask_logits = upscaled_masks
    loss = pairwise_loss(mask_logits, gt_mask, batch['lab_image'],
                         pairwise_size, pairwise_dilation)
    return loss


def critn_mse(outputs, gt_mask, batch):
    gt_mask = gt_mask.float()
    low_res_masks = outputs.pred_masks
    upscaled_masks = postprocess_masks(
        low_res_masks.squeeze(1),
        batch["reshaped_input_sizes"][0].tolist(),
        batch["original_sizes"][0].tolist()).to(gt_mask.device)
    predicted_masks = F.normalize(F.threshold(upscaled_masks, 0.0, 0))
    loss = torch.nn.MSELoss(reduction='mean')(
        predicted_masks, gt_mask.unsqueeze(1)) * 10
    return loss


def critn_med(outputs, gt_mask, batch):
    gt_mask = gt_mask.float()
    low_res_masks = outputs.pred_masks
    upscaled_masks = postprocess_masks(
        low_res_masks.squeeze(1),
        batch["reshaped_input_sizes"][0].tolist(),
        batch["original_sizes"][0].tolist()).to(gt_mask.device)
    seg_loss = DiceCELoss(sigmoid=True,
                          squared_pred=True,
                          reduction='mean')
    loss = seg_loss(upscaled_masks,
                    gt_mask.unsqueeze(1))
    return loss


def critn_sam(outputs, gt_mask: torch.Tensor, batch) -> Tensor:
    low_res_masks = outputs.pred_masks
    upscaled_masks = postprocess_masks(
        low_res_masks.squeeze(1),
        batch["reshaped_input_sizes"][0].tolist(),
        batch["original_sizes"][0].tolist()).to(gt_mask.device)
    'process upscaled masks'
    '''Compute iou by thresholding
    predicted_masks = threshold(upscaled_masks, 0.0, 0)
    predicted_masks = normalize(
                threshold(upscaled_masks, 0.0, 0))
    '''
    predicted_masks = torch.FloatTensor(torch.sigmoid(upscaled_masks))
    gt_mask = torch.LongTensor(gt_mask.unsqueeze(1))
    batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
        predicted_masks,
        gt_mask,
        mode='binary',
        threshold=0.5,
    )
    batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
    loss_iou = F.mse_loss(outputs.iou_scores.squeeze(1),
                          batch_iou, reduction='mean')
    # compute focal and dice loss
    mask_logits = upscaled_masks.flatten(1)
    gt_mask_logits = gt_mask.flatten(1).float()
    nb_masks = mask_logits.shape[0]
    loss_focal = sigmoid_focal_loss(mask_logits, gt_mask_logits, nb_masks)
    loss_dice = dice_loss(mask_logits, gt_mask_logits, nb_masks)
    return loss_iou + loss_focal * 20. + loss_dice


def get_critn(critn_name) -> Callable:
    critn_dict = {
        'mse': critn_mse,
        'med': critn_med,
        'sam': critn_sam,
        'pair': critn_pair,
    }
    return critn_dict[critn_name]
