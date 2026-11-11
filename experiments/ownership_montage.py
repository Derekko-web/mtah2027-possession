"""One explicit reference/temporal layout for local-only ownership inference."""
from PIL import Image, ImageDraw, ImageFont

SIZE = (1536, 936)
REFERENCE = (384, 0, 1152, 360)
SCENES = [(left, 420, left+512, 932) for left in (0, 512, 1024)]


def paste_fit(canvas, image, box):
    width, height = box[2]-box[0], box[3]-box[1]
    scale = min(width/image.width, height/image.height)
    resized = image.convert('RGB').resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.LANCZOS)
    canvas.paste(resized, (box[0]+(width-resized.width)//2, box[1]+(height-resized.height)//2))


def make_montage(reference, scenes):
    if len(scenes) != 3:
        raise ValueError('Ownership montage requires exactly three ordered scenes')
    canvas = Image.new('RGB', SIZE, 'white')
    paste_fit(canvas, reference, REFERENCE)
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=28)
    for image, box, label in zip(scenes, SCENES, ('EARLIER', 'MIDDLE', 'LATER')):
        draw.text((box[0]+12, 384), label, fill='black', font=font)
        paste_fit(canvas, image, box)
    return canvas
