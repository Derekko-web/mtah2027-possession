"""Local jersey statistics and homography-projected torso crops; no target labels."""

import subprocess

import numpy as np

from games import require
from starter.load_homography import project

WIDTH, HEIGHT = 960, 540
PITCH = np.array([105.3, 68.0])


def decode_frame(frame):
    """Accurate input seeking at the source's verified25fps, returning local RGB."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
               "-ss", f"{int(frame)/25:.8f}", "-i", require(["video"])[0],
               "-frames:v", "1", "-filter_threads", "1", "-vf", "scale=960:540:flags=bilinear",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode or len(result.stdout) != WIDTH*HEIGHT*3:
        raise ValueError("Local tactical frame decode failed")
    return np.frombuffer(result.stdout, np.uint8).reshape(HEIGHT, WIDTH, 3)


def weak_frame_split(n, blocks, reserved, seed=42):
    eligible = np.zeros(n, bool)
    for _, start, end in blocks:
        eligible[start:end+1] = True
    eligible[11185-250:18685+250] = False
    for window in reserved:
        eligible[max(0,window["start"]-250):min(n,window["end"]+251)] = False
    frames = np.flatnonzero(eligible & (np.arange(n)%250 == 0))
    rng = np.random.default_rng(seed)
    validation = np.sort(rng.choice(frames, 80, replace=False))
    candidates = frames[np.min(np.abs(frames[:, None]-validation[None]), axis=1) > 250]
    if len(candidates) < 240:
        raise ValueError("Not enough excluded weak training frames")
    training = np.sort(rng.choice(candidates, 240, replace=False))
    return training, validation


def isolated_players(positions, speeds):
    valid = (np.isfinite(positions).all(axis=1) & (positions[:,0] >= 10) &
             (positions[:,0] <= PITCH[0]-10) & (positions[:,1] >= 3) &
             (positions[:,1] <= PITCH[1]-3) & (speeds >= .5) & (speeds <= 12))
    distances = np.linalg.norm(positions[:,None]-positions[None],axis=2)
    distances[~np.isfinite(distances)] = np.inf
    np.fill_diagonal(distances,np.inf)
    return np.flatnonzero(valid & (distances.min(axis=1) >= 3))


def torso_bounds(matrix, position, ratio=1.):
    if not np.isfinite(position).all() or not np.isfinite(matrix).all():
        return None
    normalized = np.asarray(position)/PITCH
    x,y = project(matrix,*normalized)
    xx,xy = project(matrix,normalized[0]+1/PITCH[0],normalized[1])
    yx,yy = project(matrix,normalized[0],normalized[1]+1/PITCH[1])
    jacobian = np.array([[xx-x,yx-x],[xy-y,yy-y]])/2
    if not np.isfinite(jacobian).all(): return None
    body_height = np.linalg.svd(jacobian,compute_uv=False)[0]*1.75*ratio
    if not 4 <= body_height <= 90: return None
    x,y = x/2,y/2
    left,right = int(np.floor(x-.22*body_height)),int(np.ceil(x+.22*body_height))
    top,bottom = int(np.floor(y-.85*body_height)),int(np.ceil(y-.35*body_height))
    if left < 0 or top < 0 or right > WIDTH or bottom > HEIGHT or (right-left)*(bottom-top) < 12:
        return None
    return left,top,right,bottom


def color_statistics(crop):
    rgb = np.asarray(crop,dtype=float).reshape(-1,3)/255
    if len(rgb) < 12: return None
    grass = (rgb[:,1] > 1.15*rgb[:,0]) & (rgb[:,1] > 1.05*rgb[:,2])
    fraction = float((~grass).mean())
    rgb = rgb[~grass]
    if len(rgb) < 6: return None
    maximum,minimum = rgb.max(axis=1),rgb.min(axis=1)
    chroma = maximum-minimum
    hue = np.zeros(len(rgb))
    for channel in range(3):
        use = (rgb.argmax(axis=1) == channel) & (chroma > 1e-9)
        hue[use] = ((rgb[use,(channel+1)%3]-rgb[use,(channel+2)%3])/chroma[use]+2*channel)%6/6
    saturation = np.divide(chroma,maximum,out=np.zeros_like(chroma),where=maximum > 1e-9)
    h = np.minimum((hue*12).astype(int),11)
    s = np.minimum((saturation*3).astype(int),2)
    v = np.minimum((maximum*3).astype(int),2)
    histogram = np.bincount((h*3+s)*3+v,minlength=108)/len(rgb)
    return np.r_[histogram,rgb.mean(axis=0),rgb.std(axis=0),np.quantile(rgb,[.1,.5,.9],axis=0).ravel(),fraction].astype(np.float32)


def player_statistics(image, matrix, positions, ratio=1.):
    if image.shape != (HEIGHT,WIDTH,3): raise ValueError("Expected960x540 RGB")
    features = np.full((len(positions),124),np.nan,dtype=np.float32)
    for slot,position in enumerate(positions):
        box = torso_bounds(matrix,position,ratio)
        if box is None: continue
        left,top,right,bottom = box
        value = color_statistics(image[top:bottom,left:right])
        if value is not None: features[slot] = value
    return features
