import numpy as np
import cv2, zlib, base64, io
from PIL import Image
import json
import os


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


def get_images(folder, w, h):
    images = []
    json_data = []

    try:
        for f in os.listdir(f'{folder}/img'):
            images.append(cv2.resize(cv2.imread(f'{folder}/img/{f}', cv2.IMREAD_GRAYSCALE), (w, h)))
            with open(f'{folder}/ann/{f}.json', 'r') as file:
                json_data.append(json.load(file))
    except Exception as e:
        print(e)

    return images, json_data


