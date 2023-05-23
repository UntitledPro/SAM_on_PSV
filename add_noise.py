import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


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

t = 0
for line in lines:
    anno_info = line.split('\n')[0]
    image = str(img_path / (anno_info + '.jpg'))
    mask_in = np.array(Image.open(
        str(mask_path / (anno_info + '.png'))))
    # 图像的二值化
    # mask_in = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    _, mask_in = cv2.threshold(mask_in, np.mean(mask_in), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 设置卷积核5*5
    kernel = np.ones((7, 7), np.uint8)
    # 图像的腐蚀，默认迭代次数
    erosion = cv2.erode(mask_in, kernel)
    # 图像的膨胀
    dst = cv2.dilate(mask_in, kernel)

    plt.figure(figsize=(10, 10))
    _, ax = plt.subplots(1, 3, sharex=True)
    # 效果展示
    ax[0].imshow(mask_in)
    ax[0].set_title('original')
    ax[0].axis('off')
    # 腐蚀后
    ax[1].imshow(erosion)
    ax[1].set_title('erosion')
    ax[1].axis('off')
    # 膨胀后
    ax[2].imshow(dst)
    ax[2].set_title('dilate')
    ax[2].axis('off')

    plt.show()
    plt.close()

    t += 1
    if t > 0:
        break
