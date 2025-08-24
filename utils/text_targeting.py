# utils/text_targeting.py
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Deque
import cv2
import numpy as np
import pytesseract
import re
import time

HUD_BLOCKLIST = [
    r"Doświadczenie", r"Następny Poziom", r"Zapisano", r"Zapis gry",
    r"Załadowano", r"Zapamiętano", r"Wczytywanie", r"Gothic"
]
HUD_BLOCKLIST_RE = re.compile("|".join(HUD_BLOCKLIST), re.IGNORECASE)

@dataclass
class TextCandidate:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]          # (cx, cy)
    offset: Tuple[int, int]          # (dx, dy) relative to the screen center
    zone: str                        # 'center' | 'Q1'..'Q4'
    score: float                     # isText P(txt)
    text: Optional[str] = None       # OCR value
    conf: Optional[float] = None     # OCR confidence
    ts: float = time.time()

def _zone_label(cx: int, cy: int, W: int, H: int, inner_frac: float = 0.18) -> str:
    """center box (inner_frac width/height) vs. Qs/corners"""
    cx0, cy0 = W // 2, H // 2
    inner_w, inner_h = int(W * inner_frac), int(H * inner_frac)
    if abs(cx - cx0) <= inner_w // 2 and abs(cy - cy0) <= inner_h // 2:
        return "center"
    if cx >= cx0 and cy < cy0:
        return "Q1"
    if cx < cx0 and cy < cy0:
        return "Q2"
    if cx < cx0 and cy >= cy0:
        return "Q3"
    return "Q4"

def _mser_candidates(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """MSER → blob contours of text-like objects ; rtrn bbox."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # contrast goes brrr
    gray = cv2.equalizeHist(gray)

    mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=3000)
    regions, _ = mser.detectRegions(gray)

    boxes = []
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts)
        # object size filters
        if w < 12 or h < 10:
            continue
        ar = w / float(h)
        if ar < 0.7 or ar > 12.0:
            continue
        boxes.append((x, y, w, h))

    # non-maximum suppression on bboxes for deduplication
    boxes = _nms_boxes(boxes, iou_thresh=0.4)
    return boxes

def _nms_boxes(boxes: List[Tuple[int,int,int,int]], iou_thresh=0.4) -> List[Tuple[int,int,int,int]]:
    if not boxes: return boxes
    areas = [w*h for (x,y,w,h) in boxes]
    idxs = np.argsort(areas)[::-1].tolist()
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(boxes[i])
        x1,y1,w1,h1 = boxes[i]
        A1 = areas[i]
        rest = []
        for j in idxs:
            x2,y2,w2,h2 = boxes[j]
            iou = _iou((x1,y1,w1,h1),(x2,y2,w2,h2))
            if iou < iou_thresh:
                rest.append(j)
        idxs = rest
    return keep

def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union

def detect_text_candidates(frame_bgr: np.ndarray, limit: int = 25) -> List[Tuple[int,int,int,int]]:
    """entrypoint for text-pretend-objects – geometrical only (not OCR yet)."""
    if frame_bgr is None:
        return []
    if frame_bgr.size == 0:
        return []
    boxes = _mser_candidates(frame_bgr)
    if boxes is None:
        return []
    boxes = sorted(boxes, key=lambda b: b[3], reverse=True)[:limit]
    return boxes

def candidates_to_targets(frame_bgr: np.ndarray,
                          boxes: List[Tuple[int,int,int,int]],
                          debug_preproc=False) -> List[TextCandidate]:
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return []

    if not boxes:
        return []

    H, W = frame_bgr.shape[:2]
    cx0, cy0 = W//2, H//2
    out: List[TextCandidate] = []

    for (x,y,w,h) in boxes:
        cx, cy = x + w//2, y + h//2
        dx, dy = cx - cx0, cy - cy0
        zone = _zone_label(cx, cy, W, H)

        # heur: „score”: center proximity preference + font height ~20-40px
        height_score = np.exp(-((h-28)**2)/(2*10**2))
        center_score = np.exp(-abs(dx)/(W*0.15))
        score = 0.6*center_score + 0.4*height_score

        out.append(TextCandidate(
            bbox=(x,y,w,h),
            center=(cx,cy),
            offset=(dx,dy),
            zone=zone,
            score=float(score)
        ))
    out.sort(key=lambda c: c.score, reverse=True)
    return out

def ocr_on_candidate(frame_bgr: np.ndarray, cand: TextCandidate,
                     refine: bool = True) -> TextCandidate:
    x,y,w,h = cand.bbox
    crop = frame_bgr[y:y+h, x:x+w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # fast OCR
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang="pol")
    texts, confs, bbs = [], [], []
    for i, t in enumerate(data["text"]):
        t = t.strip()
        if not t:
            continue
        conf = float(data["conf"][i]) if data["conf"][i] != '-1' else 0.0
        bb = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        texts.append(t); confs.append(conf); bbs.append(bb)

    if not texts:
        cand.text, cand.conf = None, None
        return cand

    # max confidence
    k = int(np.argmax(confs))
    t_best, c_best, bb_best = texts[k], confs[k], bbs[k]

    # block HUDs (after OCR)
    if HUD_BLOCKLIST_RE.search(t_best):
        cand.text, cand.conf = None, None
        return cand

    # bbox refination, shoot second OCR
    if refine:
        (rx, ry, rw, rh) = bb_best
        rx = max(0, rx-2); ry = max(0, ry-2)
        rw = min(w-rx, rw+4); rh = min(h-ry, rh+4)
        sub = gray[ry:ry+rh, rx:rx+rw]
        data2 = pytesseract.image_to_data(sub, output_type=pytesseract.Output.DICT, lang="pl")
        texts2, confs2 = [], []
        for i, t in enumerate(data2["text"]):
            t = t.strip()
            if not t: continue
            c = float(data2["conf"][i]) if data2["conf"][i] != '-1' else 0.0
            texts2.append(t); confs2.append(c)
        if texts2:
            k2 = int(np.argmax(confs2))
            t_best = texts2[k2]
            c_best = confs2[k2]

    cand.text, cand.conf = t_best, float(c_best)
    return cand

class TargetHistory:
    """simple buffer for text movements (stability, float)."""
    def __init__(self, maxlen: int = 15):
        self.buf: Deque[TextCandidate] = deque(maxlen=maxlen)

    def push(self, cand: TextCandidate):
        self.buf.append(cand)

    def stable_centered(self, dx_thresh: int = 40, min_frames: int = 4) -> bool:
        if len(self.buf) < min_frames:
            return False
        xs = [abs(c.offset[0]) for c in self.buf][-min_frames:]
        return all(x <= dx_thresh for x in xs)

    def mean_dx(self) -> Optional[float]:
        if not self.buf: return None
        return float(np.mean([c.offset[0] for c in self.buf]))
