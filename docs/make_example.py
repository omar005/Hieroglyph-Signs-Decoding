"""Regenerate docs/example.png (input photo next to numbered detections).

Run from the repository root:
    python docs/make_example.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Characters_Detection import Characters_Detection  # noqa: E402

SRC = "Images/simple.jpg"
DST = "docs/example.png"

detector = Characters_Detection(SRC, R2L=False)
with tempfile.TemporaryDirectory() as tmp:
    detector.getCharacters(tmp)
annotated = detector.draw_detections()

gap = np.full((detector.img.shape[0], 12, 3), 255, np.uint8)
figure = np.hstack([cv2.imread(SRC), gap, annotated])
cv2.imwrite(DST, figure, [cv2.IMWRITE_PNG_COMPRESSION, 9])
print(f"Wrote {DST} ({len(detector.detections)} signs detected)")
