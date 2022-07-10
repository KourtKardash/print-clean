from argparse import ArgumentParser
from math import copysign
from pathlib import Path

from numpy import uint8, vectorize
from skimage.filters import threshold_local
from skimage.io import imread, imsave
from skimage.restoration import denoise_tv_chambolle

BLOCK_SIZE = 99
OFFSET = 10 / 255
DENOISE_WEIGHT = 0.03
STRENGTH_THRESHOLD = 0.02
OUTPUT_SUFFIX = '-cleaned'
OUTPUT_EXTENSION = 'png'


def compute_strength(diff: float) -> float:
    strength = min(abs(diff), STRENGTH_THRESHOLD) * 0.5 / STRENGTH_THRESHOLD
    return 0.5 + copysign(strength, diff)


def get_output_path(input_path: Path) -> Path:
    return input_path.parent / f'{input_path.stem}{OUTPUT_SUFFIX}.{OUTPUT_EXTENSION}'


args = ArgumentParser()
args.add_argument('image', help='Path to the image file')
path = Path(args.parse_args().image)

if not path.exists():
    raise Exception(f'{path} does not exist')

if not path.is_file():
    raise Exception(f'{path} is not a file')

image = imread(path, as_gray=True)
image = denoise_tv_chambolle(image, weight=DENOISE_WEIGHT)
threshold = threshold_local(image, BLOCK_SIZE, offset=OFFSET)
imsave(get_output_path(path), (vectorize(compute_strength)(image - threshold) * 255).astype(uint8))
