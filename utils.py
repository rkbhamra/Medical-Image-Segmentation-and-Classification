import numpy as np
import cv2, zlib, base64, io
from PIL import Image
import json
import os
import segmentation


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


def get_images(img_path, w, h, mendeley=False, c=0, limit=-1):
    images = []
    classes = []
    i = 0
    print('loading data...', c)

    try:
        for f in os.listdir(img_path):
            images.append(cv2.resize(cv2.imread(f'{img_path}/{f}'), (w, h)) / 255.0)
            if mendeley:
                classes.append(c)
            else:
                classes.append(int(f.replace('.png', '')[-1]))

            i += 1
            if i % 100 == 0:
                print(f'loaded {i} images')

            if i == limit:
                break
    except Exception as e:
        print(e)

    print(f'done, loaded {i} images')
    return np.array(images), np.array(classes)


def get_image(img_path, w, h):
    # _, img = get_masked_lungs(img_path, 256)
    # img = cv2.resize(img, (w, h)) / 255.0
    img = cv2.resize(cv2.imread(img_path), (w, h)) / 255.0
    return img


def get_masked_lungs(img_path, size):
    lung_filter, masked_lungs = segmentation.segmentation(img_path, size)
    return lung_filter, masked_lungs


def get_ui_output(img_path, w, h, class_type):
    color = [0, 1, 0] if class_type == 0 else [1, 0, 0] # this is red because OpenCV uses BGR
    img = get_image(img_path, w, h)
    mask, _ = get_masked_lungs(img_path, 256)
    mask = cv2.resize(mask, (w, h))
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = mask * color

    # cv2.imshow('MASK', mask)
    # cv2.imshow('ma', mask)
    # cv2.waitKey
    # print(img[256][200])
    # print(img[256][400])
    img = (img * 255).astype(np.uint8)
    # print(img[256][400])
    # print("MASK 1", mask[250][210:240])
    mask = mask.astype(np.uint8)
    # print("MASK 2", mask[250][210:240])

    # ret = cv2.addWeighted(img, 0.8, mask, 0.5, 0)#.astype(np.uint8)
    # cv2.imshow('aa', img)
    # cv2.imshow('bb', mask)
    # cv2.imshow('ee', ret)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return cv2.addWeighted(img, 1, mask, 0.8, 0)#.astype(np.uint8)#ret




