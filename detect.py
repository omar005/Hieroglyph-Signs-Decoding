"""Command-line interface for hieroglyph character detection.

Example:
    python detect.py Images/hori.jpg --direction ltr --out output/
"""

import argparse
import os
import sys

import cv2

from Characters_Detection import Characters_Detection


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect hieroglyph signs in an image and save each as a crop."
    )
    parser.add_argument("image", help="path to the input image")
    parser.add_argument(
        "--direction",
        choices=["ltr", "rtl"],
        default="ltr",
        help="reading direction of the text (default: ltr). Use rtl when the "
        "human/animal figures face right.",
    )
    parser.add_argument(
        "--out",
        default="output",
        help="folder for the cropped signs (default: output/)",
    )
    parser.add_argument(
        "--annotate",
        metavar="FILE",
        help="also save the input image with numbered boxes around each sign",
    )
    args = parser.parse_args(argv)

    try:
        detector = Characters_Detection(args.image, R2L=(args.direction == "rtl"))
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    saved = detector.getCharacters(args.out)
    print(f"Saved {len(saved)} signs to {args.out}/")

    if args.annotate:
        folder = os.path.dirname(args.annotate)
        if folder:
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(args.annotate, detector.draw_detections())
        print(f"Saved annotated image to {args.annotate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
