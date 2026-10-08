"""Hieroglyph character detection (OCR) using classical computer vision.

Finds individual signs in a photo of an inscription and saves each one as a
cropped image named ``L<line>Q<order>.jpg``:

* ``L`` - the line (row or column) the sign was assigned to, 1-based.
* ``Q`` - the sign's overall reading-order index in the image, 1-based.

Usage::

    detector = Characters_Detection("Images/hori.jpg", R2L=False)
    detector.getCharacters("output/")
"""

import os

import cv2
import numpy as np

from classes.HoughBundler import HoughBundler

# Contours smaller than this (in pixels^2) are treated as noise.
MIN_CONTOUR_AREA = 100


class Characters_Detection:
    """Detects and crops hieroglyph signs from an image.

    Args:
        inputImage: Path to the input image.
        R2L: ``True`` if the text is read right-to-left (figures face right).
            The image is then flipped horizontally so that it is processed
            as left-to-right text, and the saved crops are flipped too.
    """

    def __init__(self, inputImage, R2L=False):
        path = str(inputImage)
        self.img = cv2.imread(path)
        if self.img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if R2L:
            self.img = cv2.flip(self.img, 1)
        # Filled in by getCharacters(): one dict per saved crop.
        self.detections = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def getCharacters(self, OutputPath="output"):
        """Detect signs and save each one as a crop in ``OutputPath``.

        The folder is created if it does not exist. Existing files in it
        are not removed.

        Returns:
            The list of saved crop paths, in reading order.
        """
        edges = self._edge_map()
        boxes = self._find_boxes(edges)
        lines = self._find_separator_lines(edges)

        bundler = HoughBundler()
        vertical = bundler.chk_I_V2(lines) if lines else False

        # Sign order: distance of the sign's centre from the top-left corner.
        boxes.sort(key=lambda b: b["dist"])

        os.makedirs(OutputPath, exist_ok=True)
        self.detections = []
        saved = []
        for order, box in enumerate(boxes, start=1):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            line_no = self._line_number(box, lines, vertical)
            name = f"L{line_no}Q{order}.jpg"
            file_path = os.path.join(OutputPath, name)
            cv2.imwrite(file_path, self.img[y : y + h, x : x + w])
            saved.append(file_path)
            self.detections.append(
                {
                    "file": file_path,
                    "line": line_no,
                    "order": order,
                    "box": (x, y, w, h),
                }
            )
        return saved

    def draw_detections(self):
        """Return a copy of the image with each detected sign boxed and numbered.

        Call after :meth:`getCharacters`.
        """
        annotated = self.img.copy()
        for det in self.detections:
            x, y, w, h = det["box"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                str(det["order"]),
                (x, max(y - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated

    # ------------------------------------------------------------------ #
    # Pipeline steps
    # ------------------------------------------------------------------ #
    def _edge_map(self):
        """Grayscale -> Gaussian blur -> Canny -> 3x3 dilation."""
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 1)
        edges = cv2.Canny(blur, 100, 100)
        # Dilation thickens edges so findContours gets closed shapes.
        return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    def _find_separator_lines(self, edges):
        """Detect the carved separator lines.

        Returns a list of ``[x1, y1, x2, y2]`` lines spanning the image,
        or an empty list when the image has no separator lines.
        """
        min_line_length = min(self.img.shape[:2]) / 2
        max_line_gap = 5
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=300,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None or len(lines) == 0:
            return []
        return HoughBundler().completeLines(lines, edges)

    def _find_boxes(self, edges):
        """Find candidate signs as bounding boxes of external contours."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) <= MIN_CONTOUR_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            boxes.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cx": cx,
                    "cy": cy,
                    "dist": float(np.hypot(cx, cy)),
                }
            )
        return boxes

    @staticmethod
    def _line_number(box, lines, vertical):
        """1-based line a sign belongs to: 1 + number of separators before it."""
        number = 1
        for x1, y1, _x2, _y2 in lines:
            if vertical:
                if box["cx"] > x1:
                    number += 1
            elif box["cy"] > y1:
                number += 1
        return number
