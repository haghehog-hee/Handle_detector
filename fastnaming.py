import os
from PIL import Image
import re
IMAGE_PATH = "C:\\Tensorflow\\Dataset\\new labeling january\\problematic\\"
IMAGE_PATHS = os.listdir(IMAGE_PATH)
for PATH in IMAGE_PATHS:
    image_path = IMAGE_PATH + PATH
    im = Image.open(image_path)
    PATH = re.sub(r'1','',PATH, 1)
    PATH = re.sub(r'.jpg','.png',PATH)

    IMAGE_SAVE_PATH = IMAGE_PATH + PATH
    print(IMAGE_SAVE_PATH)
    im.save(IMAGE_SAVE_PATH)