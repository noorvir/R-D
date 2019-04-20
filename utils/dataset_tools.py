import webp
import numpy as np


def encode_webp(image, quality=100, ret="numpy"):

    pic = webp.WebPPicture.from_numpy(image)
    config = webp.WebPConfig.new(quality=quality)
    buff = pic.encode(config).buffer()

    if ret == 'numpy':
        return np.frombuffer(buff, dtype=np.int8)
    else:
        return buff


def decode_webp(data):
    if type(data) == np.ndarray:
        data = data.tobytes()

    webp_data = webp.WebPData.from_buffer(data)
    np_data = webp_data.decode()

    return np_data
