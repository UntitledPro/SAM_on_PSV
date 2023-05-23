import os
import shutil
import cv2
import monai
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from statistics import mean
from typing import Dict, List, Tuple

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import SamProcessor
from transformers import SamModel
import segmentation_models_pytorch as smp
from transformers.models.maskformer.modeling_maskformer \
    import dice_loss, sigmoid_focal_loss


def set_random_seed(seed=1):
    """Set random seed and cuda backen for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_disturbed_bounding_box(ground_truth_map: np.ndarray) -> List:
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def erode_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def dilate_mask(mask: np.ndarray,
                kernel_size: int = 3,
                iterations: int = 1) -> np.ndarray:
    kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return mask


def get_point_prompt_bymask(mask: np.ndarray) -> np.ndarray:
    """
    input_points: [nb_images, nb_predictions, nb_points_per_mask, 2]
    """
    nb = 100
    mask = erode_mask(mask)
    bg_points = np.where(mask == 0)
    bg_points = [list(i) for i in bg_points]
    bg_points = list(zip(bg_points[1], bg_points[0]))
    assert len(bg_points) > 0
    indices = np.random.choice(
        np.arange(len(bg_points)),
        size=nb, replace=False) \
        if len(bg_points) >= nb else \
        np.random.choice(
            np.arange(len(bg_points)),
            size=nb, replace=True)
    bg_points = np.array(bg_points)[indices]
    bg_label = np.zeros(nb,)

    fg_points = np.where(mask != 0)
    fg_points = [list(i) for i in fg_points]
    fg_points = list(zip(fg_points[1], fg_points[0]))
    if len(fg_points) > 0:
        indices = np.random.choice(
            np.arange(len(fg_points)),
            size=nb, replace=False) \
            if len(fg_points) >= nb else \
            np.random.choice(
                np.arange(len(fg_points)),
                size=nb, replace=True)
        fg_points = np.array(fg_points)[indices]
        fg_label = np.ones(nb,)
        pmt_points = [[np.vstack([fg_points, bg_points]).tolist()]]
        pmt_labels = [[np.hstack([fg_label, bg_label]).tolist()]]
    else:
        fg_points = []
        fg_label = []
        pmt_points = [[np.vstack([bg_points, bg_points]).tolist()]]
        pmt_labels = [[np.hstack([bg_label, bg_label]).tolist()]]
    # pmt_points = np.vstack([fg_points, bg_points])
    # pmt_labels = np.hstack([fg_label, bg_label])
    return {'input_points': pmt_points,
            'input_labels': pmt_labels}


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
    seg_loss = monai.losses.DiceCELoss(sigmoid=True,
                                       squared_pred=True,
                                       reduction='mean')
    loss = seg_loss(upscaled_masks,
                    gt_mask.unsqueeze(1))
    return loss


def critn_sam(outputs, gt_mask: torch.Tensor, batch):
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
    predicted_masks = torch.sigmoid(upscaled_masks)
    batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
        predicted_masks,
        gt_mask.unsqueeze(1),
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


def get_critn(critn_name):
    critn_dict = {
        'mse': critn_mse,
        'med': critn_med,
        'sam': critn_sam,
    }
    return critn_dict[critn_name]


def calc_metrics(pred_masks: torch.Tensor,
                 gt_mask: torch.Tensor, batch) \
        -> Tuple[List[float], List[float]]:
    """
    Args:
        pred_masks: [B, nb_preds, nb_per_preds, H, W]
        gt_mask: [B, H, W]
    """
    low_res_masks = pred_masks
    upscaled_masks = postprocess_masks(
        low_res_masks.squeeze(1),
        batch["reshaped_input_sizes"][0].tolist(),
        batch["original_sizes"][0].tolist())
    mask_prob = torch.sigmoid(upscaled_masks)
    # convert soft mask to hard mask
    mask_prob = mask_prob.cpu().detach().squeeze(1)
    sam_mask = (mask_prob > 0.5).to(torch.uint8)
    iou = mask_iou(sam_mask.numpy(), gt_mask.numpy())
    dsc = DSC(sam_mask.numpy(), gt_mask.numpy())
    return iou, dsc


def mkdir_if_not_exists(path, delete=False):
    if os.path.exists(path):
        if delete:
            shutil.rmtree(path)
            os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model, optimizer, epoch,
               save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


def resume(model, optimizer, resume_path, device):
    if not os.path.exists(resume_path):
        print(f'Cannot find {resume_path}')
        return model, optimizer, 0

    ckp = torch.load(resume_path, map_location=device)
    model_dict = ckp['model_state_dict']
    opt_dict = ckp['optimizer_state_dict']
    epoch = ckp['epoch']

    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(opt_dict)
    else:
        print('Optimizer is None')
        optimizer = opt_dict

    print('Resume from Path {} '.format(resume_path))
    print('Resume from epoch {} '.format(epoch)
          + '*' * 100)
    return model, optimizer, epoch


class PSVDataset(Dataset):
    def __init__(self,
                 root='/home/jiashuo/workspace/datasets/parking_slots/PSV dataset/',
                 split='test',
                 ds_size=1.) -> None:
        super().__init__()
        label_path = os.path.join(root, f'{split}.txt')
        with open(label_path, 'r') as f:
            samples = f.readlines()
        self.image_root = os.path.join(root, 'images', split)
        self.label_root = os.path.join(root, 'labels', split)
        self.samples = [sample.strip() for sample in samples]
        if ds_size < 0.99:
            ds_len = int(len(self.samples) * ds_size)
            random.shuffle(self.samples)
            self.samples = self.samples[:ds_len]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        sample_id = self.samples[index]
        image = cv2.imread(os.path.join(self.image_root, f'{sample_id}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.get_mask(sample_id)
        # resize
        image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (600, 600), interpolation=cv2.INTER_LINEAR)
        return {'image': image, 'label': mask}

    def get_mask(self, sample_id):
        mask_path = os.path.join(self.label_root, f'{sample_id}.png')
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        # eliminate class impact
        mask = np.bool_(mask).astype(np.uint8)
        return mask


class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        # get prompt by erosion
        prompt = get_point_prompt_bymask(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image, **prompt, return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs


def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckp_path = os.path.join(args.ckp_dir, args.ckp_name)
    resume_path = os.path.join(ckp_path, 'latest.pth')
    mkdir_if_not_exists(ckp_path, delete=False)
    epoch_start = 0
    num_epochs = args.num_epochs + 1

    'init model and processor'
    model = SamModel.from_pretrained(
        "facebook/sam-vit-huge", mirror='tuna').to(device)
    processor = SamProcessor.from_pretrained(
        "facebook/sam-vit-huge", mirror='tuna')
    criterion = get_critn(args.critn)

    'init dataset and dataloader'
    train_dataset = SAMDataset(dataset=PSVDataset(split='train',
                                                  ds_size=args.ds_size),
                               processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataset = SAMDataset(dataset=PSVDataset(split='test',
                                                 ds_size=1.0),
                              processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    'freeze image encoder and prompt encoder'
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0)

    'resume'
    if args.resume:
        model, optimizer, epoch_start = \
            resume(model, optimizer, resume_path, device)
    model.to(device)

    for epoch in range(epoch_start, num_epochs):
        if epoch % args.interval == 0:
            avg_iou, avg_dsc = test(model, test_dataloader,
                                    criterion, device)
            save_path = os.path.join(
                ckp_path,
                f'epoch{epoch}_iou{avg_iou * 100:.2f}_dsc{avg_dsc * 100:.2f}.pth')
            latest_path = os.path.join(
                ckp_path,
                'latest.pth')
            save_model(model, optimizer, epoch, save_path)
            save_model(model, optimizer, epoch, latest_path)

        epoch_losses = []
        iou = []
        dsc = []
        model.train()
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_points=batch["input_points"].to(device),
                            input_labels=batch["input_labels"].to(device),
                            multimask_output=False)
            gt_mask = batch["ground_truth_mask"].to(device)
            loss = criterion(outputs, gt_mask, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iou_e, dsc_e = calc_metrics(outputs.pred_masks.cpu().detach(),
                                        gt_mask.cpu().detach(),
                                        batch)
            epoch_losses.append(loss.item())
            iou.extend(iou_e)
            dsc.extend(dsc_e)

        print('TRAINING' + '=' * 50)
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        print(f'Mean iou: {mean(iou)}')
        print(f'Mean dsc: {mean(dsc)}')
        if epoch + 1 == num_epochs:
            save_path = os.path.join(
                ckp_path, 'latest.pth')
            save_model(model, optimizer, epoch, save_path)


@torch.no_grad()
def test(model, test_dataloader,
         criterion, device) -> Tuple[float, float]:
    model.eval()
    epoch_losses = []
    iou = []
    dsc = []
    for batch in tqdm(test_dataloader):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_points=batch["input_points"].to(device),
                        input_labels=batch["input_labels"].to(device),
                        multimask_output=False)
        gt_mask = batch["ground_truth_mask"].to(device)
        loss = criterion(outputs, gt_mask, batch)

        iou_e, dsc_e = calc_metrics(outputs.pred_masks.cpu().detach(),
                                    gt_mask.cpu().detach(),
                                    batch)
        epoch_losses.append(loss.item())
        iou.extend(iou_e)
        dsc.extend(dsc_e)

    print('TESTING' + '-' * 50)
    print(f'Mean loss: {mean(epoch_losses)}')
    print(f'Mean iou: {mean(iou)}')
    print(f'Mean dsc: {mean(dsc)}')
    return mean(iou), mean(dsc)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--ds_size', type=float, default=1.)
    parser.add_argument('--ckp_dir',
                        type=str,
                        default='work_dirs/sam_psv/')
    parser.add_argument('--ckp_name', type=str, default='sam_loss')
    parser.add_argument('--resume', action='store_false')
    parser.add_argument('--critn', type=str, default='sam')
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k:15s}: {str(v)}')
    set_random_seed(args.seed)
    finetune(args)
