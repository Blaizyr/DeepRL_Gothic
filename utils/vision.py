# utils/vision.py
import cv2
import numpy as np

def preprocess_rgb_to_obs(rgb, out_size=(84, 84), crop=None, gray=True):
    if rgb is None:
        return None
    img = rgb
    if crop is not None:
        y0,y1,x0,x1 = crop
        img = img[y0:y1, x0:x1]

    img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)

    img = img.astype(np.uint8)
    return img
