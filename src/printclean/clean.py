from argparse import ArgumentParser
from math import copysign
from pathlib import Path
from sys import exit
from typing import Optional

from numpy import vectorize
from skimage.filters import threshold_local
from skimage.io import imread, imsave
from skimage.restoration import denoise_tv_chambolle

from .perspective import fix_perspective
from .utils import float_to_uint8

BLOCK_SIZE = 99
DEFAULT_LEVEL = 10
MAX_LEVEL = 255
DENOISE_WEIGHT = 0.03
STRENGTH_THRESHOLD = 0.02
OUTPUT_SUFFIX = '-cleaned'
OUTPUT_EXTENSION = 'png'


def compute_strength(diff: float) -> float:
    strength = min(abs(diff), STRENGTH_THRESHOLD) * 0.5 / STRENGTH_THRESHOLD
    return 0.5 + copysign(strength, diff)


def get_output_path(input_path: Path) -> Path:
    return input_path.parent / f'{input_path.stem}{OUTPUT_SUFFIX}.{OUTPUT_EXTENSION}'


def run() -> None:
    args = ArgumentParser()
    args.add_argument('images', nargs='+', help='Path to the image file(s).')
    args.add_argument(
        '--level',
        nargs='?',
        default=DEFAULT_LEVEL,
        type=int,
        help=f'The cleanup threshold, a value between 0 and {MAX_LEVEL} (larger is more aggressive).',
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
    level: int = args.level

    for path in paths:
        if not path.exists():
            exit(f'{path} does not exist')
        if not path.is_file():
            exit(f'{path} is not a file')

    if not 0 <= level <= MAX_LEVEL:
        exit(f'Level is not between 0 and {MAX_LEVEL}')

    for index, path in enumerate(paths):
        print(f'Processing {path} ({index + 1} of {len(paths)})...')
        image = imread(path, as_gray=True)
        image = denoise_tv_chambolle(image, weight=DENOISE_WEIGHT)
        threshold = threshold_local(image, BLOCK_SIZE, offset=level / MAX_LEVEL)
        image = vectorize(compute_strength)(image - threshold)
        if languages:
            try:
                image = fix_perspective(image, languages)
            except ValueError as error:
                exit(error.args)
        imsave(get_output_path(path), float_to_uint8(image))

    print('Done')
