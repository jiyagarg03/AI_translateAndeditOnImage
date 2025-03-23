import numpy as np
import cv2
import easyocr
from typing import List
from paddleocr import PaddleOCR
from plugins.utils.textblock import TextBlock, adjust_text_line_coordinates
from plugins.utils.pipeline_utils import lists_to_blk_list
from plugins.ocr.manga_ocr.manga_ocr import MangaOcr
from plugins.ocr.pororo.main import PororoOcr
from plugins.utils.download import get_models, manga_ocr_data, pororo_data

manga_ocr_path = 'models/ocr/manga-ocr-base'

class OCRProcessor:
    manga_ocr_cache = None
    easyocr_cache = None
    pororo_cache = None
    paddle_ocr_cache = None

    def __init__(self):
        pass

    def process(self, img: np.ndarray, blk_list: List[TextBlock], source_language: str):
        """
        This method processes the OCR based on the language and applies the appropriate OCR method.
        """
        print(f"Processing OCR for language: {source_language}")
    
        if source_language == 'Chinese':
            return self._ocr_paddle(img, blk_list)
        elif source_language == 'Japanese':
            return self._ocr_manga(img, blk_list)
        elif source_language == 'Korean':
            return self._ocr_pororo(img, blk_list)
        elif source_language == 'English':
            return self._ocr_easyocr(img, blk_list)
        else:
            raise ValueError(f"OCR for {source_language} is not supported")

    def _ocr_paddle(self, img: np.ndarray, blk_list: List[TextBlock]):
        """PaddleOCR for Chinese text extraction."""
        if self.paddle_ocr_cache is None:
            print("Initializing PaddleOCR for Chinese text.")
            import os
            os.environ['OMP_NUM_THREADS'] = '1'
            self.paddle_ocr_cache = PaddleOCR(lang='ch')
        
        result = self.paddle_ocr_cache.ocr(img)
        print(f"PaddleOCR result: {result}")

        result = result[0]
        texts_bboxes = [tuple(coord for point in bbox for coord in point) for bbox, _ in result] if result else []
        condensed_texts_bboxes = [(x1, y1, x2, y2) for (x1, y1, x2, y1_, x2_, y2, x1_, y2_) in texts_bboxes]

        texts_string = [line[1][0] for line in result] if result else []

        print(f"Detected text boxes: {texts_bboxes}")
        print(f"Detected text: {texts_string}")
        blk_list = lists_to_blk_list(blk_list, condensed_texts_bboxes, texts_string)

        return blk_list

    def _ocr_manga(self, img: np.ndarray, blk_list: List[TextBlock]):
        """MangaOCR for Japanese text extraction."""
        if self.manga_ocr_cache is None:
            print("Loading MangaOCR model for Japanese text.")
            get_models(manga_ocr_data)
            self.manga_ocr_cache = MangaOcr(pretrained_model_name_or_path=manga_ocr_path)
        
        print(f"\nProcessing {len(blk_list)} blocks with MangaOCR:")
        for i, blk in enumerate(blk_list, 1):
            x1, y1, x2, y2 = blk.xyxy
            blk.text = self.manga_ocr_cache(img[y1:y2, x1:x2])
            print(f"Block {i:02d} [{x1},{y1},{x2},{y2}]: {blk.text}")

        return blk_list

    def _ocr_easyocr(self, img: np.ndarray, blk_list: List[TextBlock]):
        """EasyOCR for English text extraction."""
        if self.easyocr_cache is None:
            print("Initializing EasyOCR for English text.")
            self.easyocr_cache = easyocr.Reader(['en'], gpu=True)

        print(f"\nProcessing {len(blk_list)} blocks with EasyOCR:")
        for i, blk in enumerate(blk_list, 1):
            x1, y1, x2, y2 = blk.xyxy
            result = self.easyocr_cache.readtext(img[y1:y2, x1:x2], paragraph=True)
            texts = [r[1] for r in result if r is not None]
            blk.text = ' '.join(texts)
            print(f"Block {i:02d} [{x1},{y1},{x2},{y2}]: {blk.text}")

        return blk_list

    def _ocr_pororo(self, img: np.ndarray, blk_list: List[TextBlock]):
        """PororoOCR for Korean text extraction."""
        if self.pororo_cache is None:
            print("Loading PororoOCR model for Korean text.")
            get_models(pororo_data)
            self.pororo_cache = PororoOcr()

        print(f"\nProcessing {len(blk_list)} blocks with PororoOCR:")
        for i, blk in enumerate(blk_list, 1):
            x1, y1, x2, y2 = blk.xyxy
            self.pororo_cache.run_ocr(img[y1:y2, x1:x2])
            result = self.pororo_cache.get_ocr_result()
            descriptions = result['description']
            blk.text = ' '.join(descriptions)
            print(f"Block {i:02d} [{x1},{y1},{x2},{y2}]: {blk.text}")

        return blk_list
