# print-clean

This script helps prepare photographed documents for printing:

Before:

![before](/demo/demo.jpg)

After:

![after](/demo/demo-cleaned.png)

Usage:

```
python clean.py path-to-image [--lang code1 code2 ...]
```

You may provide your document’s language(s) with the `--lang` parameter, which may help fix the photo’s perspective. Currently, the script can only perform simple rotations.

You can find available languages and their codes [here](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html). You might need to specifically install your required language for the Tesseract OCR engine ([Windows](https://stackoverflow.com/a/69958671/430083), [Ubuntu](https://askubuntu.com/a/798492/1064838), [Mac OS](https://stackoverflow.com/a/60595075/430083)).

The output image will be saved as a PNG file with the suffix `-cleaned`.