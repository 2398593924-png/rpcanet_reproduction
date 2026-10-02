import numpy as np
from PIL import Image
import cv2
import os

# 废弃
imgs = []

for f in os.listdir("./datasets/images"):
    img = Image.open(os.path.join("./datasets/images", f)).convert('RGB')
    img = np.array(img)
    imgs.append(img)

imgs = np.stack(imgs, axis=0).astype(np.float32) / 255.0

mean = imgs.mean(axis = (0, 1, 2))
std = imgs.std(axis = (0, 1, 2))

print(f"mean = {mean}")
print(f"std = {std}")