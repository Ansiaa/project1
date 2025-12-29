from __future__ import annotations
import cv2
import numpy as np

def roi_crop_largest_component(bgr: np.ndarray, pad_ratio: float = 0.02) -> np.ndarray:
    """
    배경을 최대한 줄이고 물체 영역을 크게 남기는 bbox 크롭.
    - Otsu threshold + morphology로 foreground 마스크 생성
    - 가장 큰 컴포넌트의 bounding box로 crop
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # foreground가 어두운 쪽일 때를 대비해 뒤집기(평균으로 대충 판단)
    if th.mean() > 127:
        th = cv2.bitwise_not(th)

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr

    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)

    return bgr[y0:y1, x0:x1]