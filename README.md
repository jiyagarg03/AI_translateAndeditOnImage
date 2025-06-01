# AI Translate & Edit Text on Image

A Python-based AI tool that automatically detects text from an image using OCR, translates it into a desired language, and replaces the original text directly on the image. This project combines computer vision, translation, and image editing to localize images across languages.

## Description

This tool automates the process of translating images containing text — such as posters, screenshots, or manga panels. It performs text detection with OCR, translates the extracted text using an external API (like Google Translate or DeepL), and then redraws the translated text back onto the original image using OpenCV.

## Features

- Detects text regions using Tesseract OCR
- Translates text into any target language
- Removes original text background intelligently
- Replaces it with the translated text using OpenCV
- Supports multiple image formats (JPG, PNG, etc.)
- CLI-based with customizable input/output paths

## Tech Stack

- Python
- Tesseract OCR
- OpenCV
- PIL (Pillow)
- Google Translate / DeepL API (configurable)

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/jiyagarg03/AI_translateAndeditOnImage.git
cd AI_translateAndeditOnImage
