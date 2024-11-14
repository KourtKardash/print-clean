from argparse import ArgumentParser
from math import copysign
from pathlib import Path
from sys import exit
from typing import Optional

import cv2
import numpy as np

from skimage.filters import threshold_local
from skimage.restoration import denoise_tv_chambolle

from .perspective import fix_perspective
from .utils import float_to_uint8

BLURRED_WND = 21
CONTRAST_PARAM = 0.3
WINDOW_SIZE = 128

BLOCK_SIZE = 99
DEFAULT_LEVEL = 10
MAX_LEVEL = 255
DENOISE_WEIGHT = 0.03
STRENGTH_THRESHOLD = 0.02

OUTPUT_SUFFIX = '-cleaned'
OUTPUT_EXTENSION = 'png'

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def compute_strength(diff: float) -> float:
    strength = min(abs(diff), STRENGTH_THRESHOLD) * 0.5 / STRENGTH_THRESHOLD
    return 0.5 + copysign(strength, diff)


def get_output_path(input_path: Path, method: str) -> Path:
    return input_path.parent / f'{input_path.stem}{OUTPUT_SUFFIX}-{method}.{OUTPUT_EXTENSION}'


def run() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='method', help="Image processing method.")

    parser_threshold = subparsers.add_parser('threshold', help='Local threshold algorithm.')
    parser_threshold.add_argument('images', nargs='+', help='Path to the image file(s).')
    parser_threshold.add_argument(
        '--level',
        nargs='?',
        type=int,
        default=DEFAULT_LEVEL,
        help='The cleanup threshold, a value between 0 and {MAX_LEVEL} (larger is more aggressive).',
    )
    parser_threshold.add_argument(
        '--lang',
        nargs='+',
        help='Language(s) of the document. This is used to fix perspective of the photo. '
             'Use language codes from https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.',
    )

    parser_gauss = subparsers.add_parser('gauss_blur', help='Algorithm based on gaussian blur.')
    parser_gauss.add_argument('images', nargs='+', help='Path to the image file(s).')
    parser_gauss.add_argument(
        '--lang',
        nargs='+',
        help='Language(s) of the document. This is used to fix perspective of the photo. '
             'Use language codes from https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.',
    )
    args = parser.parse_args()

    paths = list(map(Path, args.images))
    languages: Optional[list[str]] = args.lang
    method: str = args.method

    for path in paths:
        if not path.exists():
            exit(f'{path} does not exist')
        if not path.is_file():
            exit(f'{path} is not a file')

    for index, path in enumerate(paths):
        print(f'Processing {path} ({index + 1} of {len(paths)})...')
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == 'gauss_blur':
            blurred = cv2.GaussianBlur(image, (BLURRED_WND, BLURRED_WND), 0)
            shadow_removed = cv2.divide(image, blurred, scale=MAX_LEVEL)
            image = adjust_gamma(shadow_removed , gamma=CONTRAST_PARAM)

            height, width = image.shape
            for y in range(0, height, WINDOW_SIZE):
                for x in range(0, width, WINDOW_SIZE):
                    window = image[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]
                    
                    mean_intensity = np.mean(window)
                    std_intensity = np.std(window)
                    
                    threshold = mean_intensity - 0.5 * std_intensity
                    binary_window = (window > threshold)            
                    image[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE][binary_window] = MAX_LEVEL
        else :
            level: int = args.level
            if not 0 <= level <= MAX_LEVEL:
                exit(f'Level is not between 0 and {MAX_LEVEL}')

            image = denoise_tv_chambolle(image, weight=DENOISE_WEIGHT)
            threshold = threshold_local(image, BLOCK_SIZE, offset=level / MAX_LEVEL)
            image = float_to_uint8(np.vectorize(compute_strength)(image - threshold))

        if languages:
            try:
                image = fix_perspective(image, languages)
            except ValueError as error:
                exit(error.args)
        cv2.imwrite(get_output_path(path, method), image)
    print('Done')