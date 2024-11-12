from argparse import ArgumentParser
from math import copysign
from pathlib import Path
from sys import exit
from typing import Optional
import cv2
import numpy as np

from .perspective import fix_perspective

BLURRED_WND = 21
CONTRAST_PARAM = 0.3
MAX_SCALE = 255
DENOISE_WEIGHT = 0.03
OUTPUT_SUFFIX = '-cleaned'
OUTPUT_EXTENSION = 'png'

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def get_output_path(input_path: Path) -> Path:
    return input_path.parent / f'{input_path.stem}{OUTPUT_SUFFIX}.{OUTPUT_EXTENSION}'


def run() -> None:
    args = ArgumentParser()
    args.add_argument('images', nargs='+', help='Path to the image file(s).')
    args.add_argument(
        '--lang',
        nargs='+',
        help='Language(s) of the document. This is used to fix perspective of the photo. '
             'Use language codes from https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.',
    )
    args = args.parse_args()

    paths = list(map(Path, args.images))
    languages: Optional[list[str]] = args.lang

    for path in paths:
        if not path.exists():
            exit(f'{path} does not exist')
        if not path.is_file():
            exit(f'{path} is not a file')

    for index, path in enumerate(paths):
        print(f'Processing {path} ({index + 1} of {len(paths)})...')
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(image, (BLURRED_WND, BLURRED_WND), 0)
        shadow_removed = cv2.divide(image, blurred, scale=MAX_SCALE)
        image = adjust_gamma(shadow_removed , gamma=CONTRAST_PARAM)

        if languages:
            try:
                image = fix_perspective(image, languages)
            except ValueError as error:
                exit(error.args)
        cv2.imwrite(get_output_path(path), image)
    print('Done')

run()