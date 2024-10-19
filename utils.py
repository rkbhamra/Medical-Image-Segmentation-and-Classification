import numpy as np
import cv2, zlib, base64, io
from PIL import Image

def base64_to_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
    return mask

def mask_to_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8), mode="L")
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format="PNG", transparency=0, optimize=0)
    bytes_enc = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes_enc)).decode("utf-8")
