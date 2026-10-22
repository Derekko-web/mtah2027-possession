"""Local rectangular-input ball detector support; no possession labels."""

import copy

import numpy as np
from PIL import Image

from native_detector import ball_scores


def letterbox_crop(image, x, y, side, width=384, height=640):
    half = side/2
    if not np.isfinite([x, y]).all() or x-half < 0 or y-half < 0 or x+half > image.width or y+half > image.height:
        return None
    length = min(width, height)
    square = image.resize((length, length), Image.Resampling.LANCZOS,
                          box=(x-half, y-half, x+half, y+half))
    canvas = Image.new('RGB', (width, height), (114, 114, 114))
    canvas.paste(square, ((width-length)//2, (height-length)//2))
    return canvas


def unletterbox_objects(objects, width=384, height=640):
    """Convert bottom-left canvas boxes to bottom-left square-crop boxes."""
    length = min(width, height)
    left, bottom = (width-length)//2, (height-length)//2
    result = copy.deepcopy(objects)
    for item in result:
        x, y, w, h = item['box']
        item['box'] = [(x*width-left)/length, (y*height-bottom)/length,
                       w*width/length, h*height/length]
    return result


def modern_scores(objects, side, diameter, width=384, height=640):
    return ball_scores(unletterbox_objects(objects, width, height), side, diameter)
