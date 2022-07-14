from numpy import uint8
from numpy.typing import NDArray

__all__ = [
    'float_to_uint8',
]


def float_to_uint8(image: NDArray[NDArray[float]]) -> NDArray[NDArray[uint8]]:
    return (image * 255).astype(uint8)
