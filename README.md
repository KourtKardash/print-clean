# print-clean

This script helps prepare photographed documents for printing:

Before:

![before](https://github.com/danmysak/print-clean/raw/main/demo/demo.jpg)

After:

![after](https://github.com/danmysak/print-clean/raw/main/demo/demo-cleaned.png)

## Installation

Install [Python](https://www.python.org/downloads/) 3.9 or higher, then run:

```
pip install printclean
```

or

```
pip3 install printclean
```

## Usage

```
printclean path-to-image [path-to-another-image ...] [--level value] [--lang code1 code2 ...]
```

You can optionally specify the cleanup’s threshold value with `--level`. The value must be an integer between 0 and 255, inclusive. Larger values produce images with fewer filled pixels. The default is 10.

You may also provide your document’s language(s) with the `--lang` parameter, which may help fix the input image’s perspective. Currently, the script can only perform simple rotations.

You can find available languages and their codes [here](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html). You might need to specifically install your required language for the Tesseract OCR engine ([Windows](https://stackoverflow.com/a/69958671/430083), [Ubuntu](https://askubuntu.com/a/798492/1064838), [Mac OS](https://stackoverflow.com/a/60595075/430083)).

The output image will be saved as a PNG file with the suffix `-cleaned`.