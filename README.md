# print-clean

This script helps prepare photographed documents for printing:

Before:

![before](https://github.com/danmysak/print-clean/raw/main/demo/demo_1.jpg)
![before](https://github.com/danmysak/print-clean/raw/main/demo/demo_2.jpg)

After:

![after](https://github.com/danmysak/print-clean/raw/main/demo/demo_1-cleaned-threshold.png)
![after](https://github.com/danmysak/print-clean/raw/main/demo/demo_2-cleaned-gauss_blur.png)

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
printclean <path-to-image> [<path-to-another-image> ...] [--method threshold|gauss_blur] [--level value] [--lang <code1> <code2> ...]
```

You can provide one or more paths to image files. The script will clean them up and save the results in the same directory.

You can specify the cleanup method using the `--method` parameter. The available methods are `threshold` and `gauss_blur`, with the default being `gauss_blur`.

If the `--method` is set to `threshold`, you can optionally specify the threshold value for cleanup using the `--level` parameter. The value must be an integer between 0 and 255 (inclusive). Larger values produce images with fewer filled pixels. The default value is 10.

Additionally, you may specify the language(s) of your document using the `--lang` parameter. This can help the script improve perspective correction for the input image. Note that the script currently supports only simple rotations for perspective correction.

You can find the available languages and their corresponding codes [here](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html). Depending on your system, you may need to install the required language files for the Tesseract OCR engine: ([Windows](https://stackoverflow.com/a/69958671/430083), [Ubuntu](https://askubuntu.com/a/798492/1064838), [Mac OS](https://stackoverflow.com/a/60595075/430083)).

The output image will be saved as a PNG file with the suffix `-cleaned`.