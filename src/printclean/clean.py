from argparse import ArgumentParser
from math import copysign
from pathlib import Path
from sys import exit
from typing import Optional

import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray
from skimage.filters import threshold_local
from skimage.restoration import denoise_tv_chambolle

from .perspective import fix_perspective
from .utils import float_to_uint8

LOCAL_THRESHOLD_METHOD = 'threshold'
GAUSSIAN_BLUR_METHOD = 'gauss_blur'
DEFAULT_METHOD = GAUSSIAN_BLUR_METHOD

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


def get_output_path(input_path: Path, method: str) -> Path:
    return input_path.parent / f'{input_path.stem}{OUTPUT_SUFFIX}-{method}.{OUTPUT_EXTENSION}'


def local_threshold(image: NDArray[NDArray[uint8]], level: int) -> NDArray[NDArray[float]]:
    image = denoise_tv_chambolle(image, weight=DENOISE_WEIGHT)
    threshold = threshold_local(image, BLOCK_SIZE, offset=level / MAX_LEVEL)
    return np.vectorize(compute_strength)(image - threshold)


def adjust_gamma(image: NDArray[NDArray[uint8]], gamma: float) -> NDArray[NDArray[float]]:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma for i in np.arange(0, 256)])
    return cv2.LUT(image, table)


def compute_strength(diff: float) -> float:
    strength = min(abs(diff), STRENGTH_THRESHOLD) * 0.5 / STRENGTH_THRESHOLD
    return 0.5 + copysign(strength, diff)


def gaussian_blur(image: NDArray[NDArray[uint8]]) -> NDArray[NDArray[float]]:
    blurred = cv2.GaussianBlur(image, (BLURRED_WND, BLURRED_WND), 0)
    shadow_removed = cv2.divide(image, blurred, scale=MAX_LEVEL)
    image = adjust_gamma(shadow_removed, gamma=CONTRAST_PARAM)

    height, width = image.shape
    for y in range(0, height, WINDOW_SIZE):
        for x in range(0, width, WINDOW_SIZE):
            window = image[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]
            mean_intensity = np.mean(window)
            std_intensity = np.std(window)
       
            threshold = mean_intensity - 0.5 * std_intensity
            binary_window = (window > threshold)            
            image[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE][binary_window] = 1

    return image


def run() -> None:
    args = ArgumentParser()
    args.add_argument('images', nargs='+', help='Path to the image file(s).')
    args.add_argument(
        '--method',
        choices=[LOCAL_THRESHOLD_METHOD, GAUSSIAN_BLUR_METHOD],
        default=DEFAULT_METHOD,
        help='The cleanup method to use.',
    )
    args.add_argument(
        '--level',
        nargs='?',
        default=DEFAULT_LEVEL,
        type=int,
        help=f'The cleanup threshold, a value between 0 and {MAX_LEVEL} '
             f'(larger is more aggressive; used only with "{LOCAL_THRESHOLD_METHOD}" method).',
    )
    args.add_argument(
        '--lang',
        nargs='+',
        help='Language(s) of the document. This is used to fix perspective of the photo. '
             'Use language codes from https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.',
    )
    args = args.parse_args()

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
        image = cv2.cvtColor(cv2.imread(str(path.resolve())), cv2.COLOR_BGR2GRAY)
        if method == LOCAL_THRESHOLD_METHOD:
            level: int = args.level
            if not 0 <= level <= MAX_LEVEL:
                exit(f'Level is not between 0 and {MAX_LEVEL}')
            cleaned = local_threshold(image, level)
        elif method == GAUSSIAN_BLUR_METHOD:
            cleaned = gaussian_blur(image)
        else:
            raise NotImplementedError(f'Method {method} is not implemented')

        if languages:
            try:
                cleaned = fix_perspective(cleaned, languages)
            except ValueError as error:
                exit(error.args)
        cv2.imwrite(str(get_output_path(path, method).resolve()), float_to_uint8(cleaned))

    print('Done')
