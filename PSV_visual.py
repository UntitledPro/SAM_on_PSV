from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path
from pycocotools.coco import COCO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import json
from copy import deepcopy
from typing import List
from PIL import Image

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def useless_func():
    'dataloader init'
    # from mmcv import Config
    # from mmdet.datasets import build_dataset, build_dataloader
    # from mmdet.utils import replace_cfg_vals, update_data_root
    # cfg_path = 'ablation_cfgs/_base_/datasets/coco_detection_sam.py'
    # cfg = Config.fromfile(cfg_path)
    # cfg = replace_cfg_vals(cfg)
    # update_data_root(cfg)
    # train_dataloader_default_args = dict(
    #     samples_per_gpu=2,
    #     workers_per_gpu=2,
    #     num_gpus=1,
    #     dist=False,
    #     seed=0,
    #     persistent_workers=False)

    # train_loader_cfg = {
    #     **train_dataloader_default_args,
    #     **cfg.data.get('train_dataloader', {})
    # }
    # dataset = build_dataset(cfg.data.train)
    # dataloader = build_dataloader(dataset, **train_loader_cfg)

    'image visualization'
    # key = next(iter(coco_noisy.imgToAnns.keys()))
    # key = 2009000536
    # ann = coco_noisy.imgToAnns[key]
    # tmp_img_path = image_path / coco_noisy.loadImgs(key)[0]['file_name']
    # image = cv2.imread(str(tmp_img_path))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # input_box = ann[0]['bbox']
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_box(input_box, plt.gca())
    # plt.axis('off')
    # plt.show()
    pass


def show_mask(mask, ax, random_color=None):
    color_value = {
        'yellow': np.array([255 / 255, 255 / 255, 0 / 255, 0.7]),
        'blue': np.array([0 / 255, 0 / 255, 255 / 255, 0.7]),
    }
    if random_color is None:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    elif random_color in color_value.keys():
        color = color_value[random_color]
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=0.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=0.25)


def show_box(box, ax, color='green', lw=2):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color,
                 facecolor=(0, 0, 0, 0), lw=lw))


def coco2xyxy(box: torch.Tensor):
    """
    Args:
        box: (*, 4) tensor in xywh format
    Returns:
        (*, 4) tensor in xyxy format
    """
    assert box.shape[-1] == 4
    x, y, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy2coco(box: torch.Tensor):
    """
    Args:
        box: (*, 4) tensor in xyxy format
    Returns:
        (*, 4) tensor in xywh coco format
    """
    assert box.shape[-1] == 4
    x, y, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return torch.stack([x, y, x2 - x, y2 - y], dim=-1)


def mask2bbox(masks: torch.Tensor) -> torch.Tensor:
    """convert mask to bbox
    Args:
        mask: (N, H, W) tensor, bool
            (nb_bboxes) x (nb_per_box) x H x W
    Returns:
        bbox: (N, 4) tensor in xyxy format
    """
    N, _, _ = masks.shape
    bbox = torch.zeros((N, 4), dtype=torch.int32)
    for idx in range(N):
        mask = masks[idx]
        horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
        vertical_indicies = torch.where(torch.any(mask, dim=1))[0]
        # w, h should be at least 1
        if horizontal_indicies.shape[0] and vertical_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        bbox[idx] = torch.tensor([x1, y1, x2, y2])
    return bbox


def bbox_iou(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Compute iou of bbox1[i] and bbox2[i] in xyxy format
    Args:
        bbox1: (N, 4) tensor in xyxy format
        bbox2: (N, 4) tensor in xyxy format
    returns:
        iou: (N,) tensor
            iou of bbox1[i] and bbox2[i] in xyxy format
    """
    N = bbox1.shape[0]
    assert N == bbox2.shape[0]
    iou = torch.zeros((N,))
    for idx in range(N):
        box1 = bbox1[idx]
        box2 = bbox2[idx]
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            iou[idx] = 0
        else:
            area = w * h
            iou[idx] = area / (area1 + area2 - area)
    return iou


def select_bbox(new_bbox: torch.Tensor,
                input_bbox: torch.Tensor,
                iou_pred: torch.Tensor) -> torch.Tensor:
    """select bbox from multi bboxes for each instance based on iou,
    the one having the biggest iou with input_bbox is selected

    Args:
        new_bbox: xyxy format, [nb_bboxes, nb_per_box, 4]
        input_bbox: xyxy format, [nb_bboxes, 4]
        iou_pred: [nb_bboxes, nb_per_box]
    Return:
        bbox: xyxy format
    """
    nb_bbox, nb_per_box, _ = new_bbox.shape
    selected_bbox = torch.zeros((nb_bbox, 4), dtype=torch.int32)
    selected_iou_pred = torch.zeros((nb_bbox,), dtype=torch.float32)
    for idx in range(nb_bbox):
        iou = bbox_iou(new_bbox[idx],
                       input_bbox[idx].repeat(nb_per_box, 1))
        selected_bbox[idx] = new_bbox[idx][iou.argmax()]
        selected_iou_pred[idx] = iou_pred[idx][iou.argmax()]
    return selected_bbox, selected_iou_pred


def sort_bbox(new_bbox: torch.Tensor,
              iou_pred: torch.Tensor) -> torch.Tensor:
    """sort int(nb_per_box) bboxes based on iou_pred for every instance,

    Args:
        new_bbox: xyxy format, [nb_bboxes, nb_per_box, 4]
        iou_pred: [nb_bboxes, nb_per_box]
    Return
        sorted_bbox: xyxy format, [nb_bboxes, nb_per_box, 4],
            sorted by iou_pred, descending
    """
    nb_bbox, nb_per_box, _ = new_bbox.shape
    sorted_bbox = torch.zeros((nb_bbox, nb_per_box, 4), dtype=torch.int32)
    sorted_iou_pred = torch.zeros((nb_bbox, nb_per_box), dtype=torch.float32)
    for idx in range(nb_bbox):
        sorted_idx = iou_pred[idx].argsort(descending=True)
        sorted_bbox[idx] = new_bbox[idx][sorted_idx]
        sorted_iou_pred[idx] = iou_pred[idx][sorted_idx]
    return sorted_bbox, sorted_iou_pred


def main(ori_anno, image_path, all=False, count_max=-1, new_ann_path=None):
    coco_noisy = COCO(ori_anno)
    device = "cuda:4"

    sam_checkpoint = "tools/SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    count = 0
    new_ann = []
    for key in coco_noisy.imgToAnns.keys():
        ann = coco_noisy.imgToAnns[key]
        tmp_img_path = image_path / coco_noisy.loadImgs(key)[0]['file_name']
        image = cv2.imread(str(tmp_img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_boxes_xywh = torch.tensor(
            [instance_ann['bbox'] for instance_ann in ann],
            device=device)
        input_boxes = coco2xyxy(input_boxes_xywh)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            input_boxes, image.shape[:2])
        # masks: [nb_bboxes, nb_per_box, H, W]
        masks, iou_pred, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )
        nb_bboxes, nb_per_box, H, W = masks.shape
        # get bbox in xyxy format [nb_bboxes * nb_per_box, H, W]
        new_bbox = mask2bbox(masks.view(-1, H, W))
        if all:
            selected_bbox, selected_iou_pred \
                = sort_bbox(new_bbox.view(nb_bboxes, nb_per_box, 4),
                            iou_pred)
        else:
            # select one bbox from multi boxes for each instance
            selected_bbox, selected_iou_pred = select_bbox(
                new_bbox.view(nb_bboxes, nb_per_box, 4),
                input_boxes, iou_pred)
        'update noisy/clean annotations'
        # change corrected_bbox to coco xywh format
        selected_bbox_xywh = xyxy2coco(selected_bbox)

        corrected_ann = deepcopy(ann)
        for ann_idx in range(len(ann)):
            corrected_ann[ann_idx]['bbox'] = \
                selected_bbox_xywh[ann_idx].cpu().numpy().tolist()
            corrected_ann[ann_idx]['score'] = \
                selected_iou_pred[ann_idx].cpu().numpy().tolist()
        new_ann.extend(corrected_ann)
        color = ['yellow', 'blue', 'purple', 'pink', 'orange']
        # iou between selected bbox and input bbox
        # iou_score = bbox_iou(selected_bbox, input_boxes)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks.view(-1, H, W):
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box in input_boxes:
            show_box(box.cpu().numpy(), plt.gca(), 'red', 0.5)
        for _, boxes in enumerate(selected_bbox):
            for idx, box in enumerate(boxes):
                show_box(box.cpu().numpy(), plt.gca(),
                         color[idx], 1 - idx / 5)
            # box = box.cpu().numpy()
            # x = (box[0] + box[2]) / 2
            # y = (box[1] + box[3]) / 2
            # plt.text(x, y,
            #          str(round(float(iou_score[idx].
            #                    cpu().numpy()),
            #                    3)),
            #          fontsize=12)
        plt.axis('off')
        plt.show()
        plt.close()

        if count_max > 0:
            count += 1
            if count > 20:
                break

    # write new annotation
    if new_ann_path:
        with open(new_ann_path, 'w') as f:
            json.dump(new_ann, f)


def show_slots(coords, ax, marker_size=5):
    """coord: x1,y1 x2,y2 x3,y3 x4,y4"""
    for coord in coords:
        x1, y1, x2, y2, x3, y3, x4, y4 = coord
        ax.scatter([x1, x2, x3, x4], [y1, y2, y3, y4], s=marker_size, c='red')


def get_slot_coords(coords: List) -> List:
    """
    coords: [x1,y1 x2,y2 x3,y3 x4,y4, x1,y1 x2,y2 x3,y3 x4,y4, ....]
    slot_coords: [nb_slots, 8]
    """
    assert len(coords) % 9 == 0
    nb_slots = len(coords) // 9
    slot_coords = []
    for i in range(nb_slots):
        if int(coords[i * 9]) >= 0:
            coord = map(int, coords[i * 9 + 1: (i + 1) * 9])
            slot_coords.append(coord)
        else:
            print(coords[i * 9])
    return slot_coords


def mask_iou(mask1, mask2):
    """Calculate the extent to which mask2 contains mask1"""
    mask1 = mask1.reshape(-1).astype(int)
    mask2 = mask2.reshape(-1).astype(int)
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    return intersection / union


def DSC(mask1, mask2):
    """Dice similarity coefficient"""
    mask1 = mask1.reshape(-1).astype(int)
    mask2 = mask2.reshape(-1).astype(int)
    intersection = np.sum(mask1 * mask2)
    return 2 * intersection / (np.sum(mask1) + np.sum(mask2))


if __name__ == "__main__":
    print('Start relabeling')
    device = "cuda:5"
    color = ['yellow', 'blue', 'purple', 'pink', 'orange']

    sam_checkpoint = Path("~/workspace/sam_vit_h_4b8939.pth").expanduser()
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    root_path = Path(
        '~/workspace/datasets/parking_slots/PSV dataset').expanduser()
    mode = 'train'
    img_path = root_path / 'images' / mode
    mask_path = root_path / 'labels' / mode
    anno_path = root_path / f'{mode}.txt'
    relabel_path = root_path / 'appendix' / 'relabel' / mode
    visual_path = root_path / 'appendix' / 'visual' / mode

    with open(anno_path, 'r') as f:
        lines = f.readlines()
    t = 1
    iou_collection = []
    dice_collection = []
    cls_collection = [[], [], [], [], []]
    for line in lines:
        anno_info = line.split('\n')[0]
        image = cv2.imread(
            str(img_path / (anno_info + '.jpg')))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_in = np.array(Image.open(
            str(mask_path / (anno_info + '.png'))))
        cls = mask_in.max()
        _, mask_in = cv2.threshold(mask_in,
                                   np.mean(mask_in), 1,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        erosion = mask_in
        temp = Path('./data')
        # temp = Path('./data/normal')
        # erosion = cv2.erode(mask_in, kernel)
        # temp = Path('./data/erosion')
        # erosion = cv2.dilate(mask_in, kernel)
        # temp = Path('./data/dilation')

        fg_points = np.where(erosion != 0)
        fg_points = [list(i) for i in fg_points]
        fg_points = list(zip(fg_points[1], fg_points[0]))
        nb_fg = min(len(fg_points), 100)
        nb_fg = len(fg_points)
        fg_points = np.array(fg_points)[
            np.random.choice(
                np.arange(len(fg_points)),
                size=nb_fg,
                replace=False)]
        fg_label = np.ones(nb_fg,)

        bg_points = np.where(erosion == 0)
        bg_points = [list(i) for i in bg_points]
        bg_points = list(zip(bg_points[1], bg_points[0]))
        nb_bg = min(len(bg_points), 200)
        bg_points = np.array(bg_points)[
            np.random.choice(
                np.arange(len(bg_points)),
                size=nb_bg,
                replace=False)]
        bg_label = np.zeros(nb_bg,)

        predictor.set_image(image)
        try:
            mask_out, iou_pred, _ = predictor.predict(
                point_coords=np.vstack([fg_points, bg_points]),
                point_labels=np.hstack([fg_label, bg_label]),
                mask_input=None,
                multimask_output=False,
            )
        except Exception:
            continue
        mask_out_t = mask_out[0].astype(np.uint8)
        mask_out_t = cv2.erode(mask_out_t, kernel)
        # mask_out_t = cv2.erode(mask_out_t, kernel)
        # mask_out_t = cv2.dilate(mask_out_t, kernel)
        mask_out_t = cv2.dilate(mask_out_t, kernel)
        mask_out = mask_out_t
        iou = mask_iou(np.bool_(mask_in),
                       np.bool_(mask_out))
        dice = DSC(np.bool_(mask_in),
                   np.bool_(mask_out))
        iou_collection.append(iou)
        dice_collection.append(dice)
        cls_collection[cls - 1].append(iou)
        t += 1
        # if cls == t:
        #     t += 1
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(image)
        #     show_mask(erosion, plt.gca(), random_color=color[0])
        #     plt.axis('off')
        #     plt.show()
        #     plt.savefig(temp / (anno_info + f'{cls}_ori.png'))
        #     plt.close()

        if t % 10 == 0:
            print(len(iou_collection))
            print('IoU: ', np.mean(iou_collection))
            print('DSC: ', np.mean(dice_collection))
            print('Cls IoU: ', [np.mean(i) for i in cls_collection])
            # break
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(mask_out, plt.gca(), random_color=color[1])
        # # show_points(np.vstack([fg_points, bg_points]),
        # #             np.hstack([fg_label, bg_label]),
        # #             plt.gca(), marker_size=100)
        # plt.title(f'IOU: {iou:.2f}')
        # plt.axis('off')
        # plt.show()
        # plt.savefig(temp / (anno_info + '_rel.png'))
        # plt.close()

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(mask_out_t, plt.gca(), random_color=color[0])
        # plt.axis('off')
        # plt.show()
        # plt.savefig(temp / (anno_info + '_ori.png'))
        # plt.close()
