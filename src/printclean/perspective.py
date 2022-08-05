from dataclasses import dataclass
from math import atan, pi
from typing import Optional

from numpy.typing import NDArray
from pytesseract import get_languages, image_to_data, Output
from skimage.color import gray2rgb
from skimage.transform import rotate

from .utils import float_to_uint8

__all__ = ['fix_perspective']

LANG_DELIMITER = '+'


@dataclass
class Word:
    left: int
    top: int
    width: int
    height: int
    text: str


Line = list[Word]


def extract_lines(data: dict[str, list[int | str]]) -> list[Line]:
    last_line_id = None
    lines: list[Line] = []
    for index, text in enumerate(data['text']):
        line_id = tuple(data[field][index] for field in ('level', 'block_num', 'par_num', 'line_num'))
        if line_id != last_line_id:
            last_line_id = line_id
            lines.append([])
        if text != '':
            lines[-1].append(Word(
                left=data['left'][index],
                top=data['top'][index],
                width=data['width'][index],
                height=data['height'][index],
                text=text,
            ))
    return [line for line in lines if line]


def get_left(word: Word) -> float:
    return word.left


def get_right(word: Word) -> float:
    return word.left + word.width


def get_middle_y(word: Word) -> float:
    return word.top + word.height / 2


def compute_average_slope(lines: list[Line]) -> Optional[float]:
    dx = 0
    dy = 0
    for words in lines:
        if len(words) > 1:
            first, last = words[0], words[-1]
            dx += get_right(last) - get_left(first)
            dy += get_middle_y(last) - get_middle_y(first)
    return dy / dx if dx > 0 else None


def slope_to_degrees(slope: float) -> float:
    return atan(slope) * 180 / pi


# Currently, only rotations are supported
def fix_perspective(image: NDArray[NDArray[float]], languages: list[str]) -> NDArray[NDArray[float]]:
    supported_languages = get_languages()
    if languages:
        for code in languages:
            if code not in supported_languages:
                raise ValueError(f'Language with the code "{code}" is not supported '
                                 f'by the Tesseract OCR engine or is not installed')
    data = image_to_data(gray2rgb(float_to_uint8(image)), lang=LANG_DELIMITER.join(languages), output_type=Output.DICT)
    lines = extract_lines(data)
    slope = compute_average_slope(lines)
    return rotate(image, slope_to_degrees(slope), resize=True, cval=1) if slope is not None else image
