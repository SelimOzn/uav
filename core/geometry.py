import math


Box = tuple[float, float, float, float]

def box_center(box: Box) -> tuple[float, float]:
    x, y, w, h = box
    return x+w*0.5, y+h*0.5

def box_area(box: Box) -> float:
    return max(0.0, box[2]) * max(0.0, box[3])

def iou(a: Box | None, b: Box | None) -> float:
    if a is None or b is None:
        return 0.0

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, by2), min(ay2, by2)

    iw ,ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0

def center_distance(a: Box, b: Box) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)

def clip_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h= box
    x = min(max(0.0, x), float(width-1))
    y = min(max(0.0, y), float(height-1))
    w = min(max(1.0, w), float(width) - x)
    h = min(max(1.0, h), float(height) - y)

    return x, y, w, h

def expand_box(box: Box, scale: float, width: int, height:int) -> Box:
    cx, cy = box_center(box)
    w = max(1.0, box[2]*scale)
    h = max(1.0, box[3]*scale)
    return clip_box((cx-w*0.5, cy-h*0.5,w,h), width, height)

