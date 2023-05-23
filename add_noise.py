import os
import numpy as np
from numpy import ndarray
from statistics import mean
from typing import Dict, List
import torch
from torch import device as tdevice
from finetune import dilate_mask, mask_iou, PSVDataset

os.environ["CUDA_VISIBLE_DEVICES"] = ""
color_list: List = [
    # Yellow
    np.array([1, 1, 0, 0.3]),
    # Blue
    np.array([0, 0, 1, 0.3]),
    # Purple
    np.array([0.5, 0, 0.5, 0.3]),
    # Red
    np.array([1, 0, 0, 0.3]),
    # Green
    np.array([0, 1, 0, 0.3]),
    # Orange
    np.array([1, 0.5, 0, 0.3]),
    # Pink
    np.array([1, 0, 1, 0.3]),
    # Brown
    np.array([0.5, 0.25, 0, 0.3]),
    # Gray
    np.array([0.5, 0.5, 0.5, 0.3]),
    # Black
    np.array([0, 0, 0, 0.3]),
    # Teal
    np.array([0, 0.5, 0.5, 0.3]),
    # Navy
    np.array([0, 0, 0.5, 0.3])
]


def show_mask(mask, ax, mask_color=None, random_color=False):
    if mask_color is not None:
        color = mask_color
    else:
        if random_color:
            color = np.concatenate(
                [np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    if issubclass(type(mask), torch.Tensor):
        mask = mask.cpu().numpy()
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


device: tdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iou: List[float] = []

train_dataset = PSVDataset(split='train', ds_size=1.)
for idx in range(len(train_dataset)):
    item: Dict[str, ndarray] = train_dataset[idx]
    image: ndarray = item['image']
    gt_mask: ndarray = item['label']
    noise_mask: ndarray = dilate_mask(gt_mask, 2, 6)
    iou_i: List[float] = mask_iou(gt_mask[None, :], noise_mask[None, :])
    iou.extend(iou_i)

    if idx % 100 == 0:
        print(f"IOU: {mean(iou):.3f}")
    # fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    # fig.suptitle(f"IOU: {iou_i[0]:.3f}", fontsize=20)

    # axes[0].imshow(image)
    # axes[0].axis('off')

    # axes[1].imshow(image)
    # axes[1].axis('off')
    # show_mask(gt_mask, axes[1], mask_color=color_list[0])

    # axes[2].imshow(image)
    # axes[2].axis('off')
    # show_mask(noise_mask, axes[2], mask_color=color_list[1])
    # plt.show()
