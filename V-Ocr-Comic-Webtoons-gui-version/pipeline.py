import os
import cv2
import shutil
import numpy as np
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import gc
import torch.cuda
import json

from plugins.detection import TextBlockDetector
from plugins.ocr.ocr import OCRProcessor
from plugins.translator import Translator
from plugins.utils.textblock import TextBlock, sort_blk_list
from plugins.rendering.render import draw_text, get_best_render_area
from plugins.utils.pipeline_utils import inpaint_map, get_config, generate_mask, get_language_code, set_alignment, is_directory_empty
from plugins.utils.translator_utils import get_raw_translation, get_raw_text, format_translations
from plugins.utils.archives import make
from plugins.inpainting.schema import Config
import concurrent.futures
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from queue import Queue
from threading import Lock

def filter_unwanted_text(blk_list: List[TextBlock], unwanted_strings: List[str]):
    """
    Filter out blocks that contain unwanted strings, but preserve blocks that only contain
    punctuation marks like !, ?, etc.
    """
    filtered_blk_list = []
    for blk in blk_list:
        # Define a set of allowed punctuation-only patterns
        punctuation_only = {'!', '!!', '!!!', '?', '??', '???', '!?', '?!', '?!?', '!?!'}
        
        # Check if the block text is just punctuation
        if blk.text.strip() in punctuation_only or blk.translation.strip() in punctuation_only:
            filtered_blk_list.append(blk)
            continue
            
        # Otherwise, apply the normal filtering
        if any(unwanted in blk.translation for unwanted in unwanted_strings) or \
           any(unwanted in blk.text for unwanted in unwanted_strings):
            print(f"Skipping block with unwanted text: {blk.text} / {blk.translation}")
        else:
            filtered_blk_list.append(blk)
    return filtered_blk_list

class ComicTranslatePipeline:
    def __init__(self):
        self.block_detector_cache = None
        self.inpainter_pool = None
        self.inpainter_lock = Lock()
        self.free_inpainters = Queue()
        # self.inpainter_cache = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.ocr = OCRProcessor()
        self.image_files = []  # List to hold paths to images
        
        # # Set this for better memory management
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            # Optional: Set memory allocation to TensorFloat32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    def initialize_inpainter_pool(self, num_workers):
        """Initialize a pool of inpainter instances with memory management"""
        InpainterClass = inpaint_map['LaMa']
        self.inpainter_pool = []
        
        # Clear memory before creating new instances
        self.clear_gpu_memory()
        
        # Create instances one at a time with memory clearing between each
        for _ in range(num_workers):
            try:
                inpainter = InpainterClass(self.device)
                self.inpainter_pool.append(inpainter)
                self.free_inpainters.put(inpainter)
                # Clear memory after each successful creation
                self.clear_gpu_memory()
            except RuntimeError as e:
                print(f"Warning: Could only initialize {len(self.inpainter_pool)} inpainters due to memory constraints")
                break
        
        if not self.inpainter_pool:
            print("Warning: Falling back to single inpainter mode due to memory constraints")
            self.inpainter_pool = [InpainterClass(self.device)]
            self.free_inpainters.put(self.inpainter_pool[0])

    def get_inpainter(self):
        """Get a free inpainter from the pool"""
        with self.inpainter_lock:
            return self.free_inpainters.get()

    def release_inpainter(self, inpainter):
        """Release an inpainter back to the pool"""
        with self.inpainter_lock:
            self.free_inpainters.put(inpainter)

    def visualize_text_blocks(self, image, blk_list):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1)
        ax.imshow(image_rgb)
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple']
        
        for idx, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            color = colors[idx % len(colors)]  
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        plt.axis('off')  
        plt.show()

    def detect_blocks(self, image):
        if self.block_detector_cache is None:
            self.block_detector_cache = TextBlockDetector(
                'models/detection/custom-bubble-detector-types.pt',
                'models/detection/comic-text-segmenter.pt', 
                'models/detection/manga-text-detector.pt', 
                self.device
            )
        blk_list = self.block_detector_cache.detect(image)
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            block_region = image[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray_region = cv2.cvtColor(block_region, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to separate text from background
            _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count white and black pixels
            white_pixels = cv2.countNonZero(binary)
            black_pixels = binary.size - white_pixels

            # Determine if text is dark or light based on pixel count
            if white_pixels > black_pixels:
                # Text is likely dark on light background
                blk.font_color = (0, 0, 0)  # Black
                blk.outline_color = (255, 255, 255)  # White

            else:
                # Text is likely light on dark background
                blk.font_color = (255, 255, 255)  # White
                blk.outline_color = (0, 0, 0)  # Black

            # Calculate average color of the background
            avg_color = np.mean(block_region, axis=(0, 1))  
            avg_color = tuple(map(int, avg_color))

            blk.bg_color = avg_color
        return blk_list

    def manual_inpaint(self, image, mask):
        """Use an inpainter from the pool to process the image with memory management"""
        inpainter = self.get_inpainter()
        try:
            config = Config(hd_strategy="Original")
            # Clear memory before processing
            self.clear_gpu_memory()
            inpaint_input_img = inpainter(image, mask, config)
            inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
            return inpaint_input_img
        finally:
            self.release_inpainter(inpainter)
            # Clear memory after processing
            self.clear_gpu_memory()


    def OCR_image(self, image, blk_list, source_lang):
        """Process image blocks for OCR with preprocessing for better accuracy"""
        for blk in blk_list:
            # Extract the region of interest for this text block
            x1, y1, x2, y2 = blk.xyxy
            roi = image[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to handle different lighting conditions
            binary = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 2
            )
            
            # Denoise the image
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Optional: Scale the image (can help with small text)
            scale_factor = 2
            scaled = cv2.resize(denoised, None, fx=scale_factor, fy=scale_factor, 
                              interpolation=cv2.INTER_CUBIC)
            
            # Create a temporary image with the processed block
            temp_img = image.copy()
            temp_img[y1:y2, x1:x2] = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            
            # Process with OCR
            self.ocr.process(temp_img, [blk], source_lang)
            
            # If text was not detected, try with the scaled version
            if not blk.text or len(blk.text.strip()) == 0:
                temp_scaled = image.copy()
                scaled_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
                temp_scaled[y1:y2, x1:x2] = cv2.resize(scaled_bgr, 
                                                     (x2-x1, y2-y1), 
                                                     interpolation=cv2.INTER_AREA)
                self.ocr.process(temp_scaled, [blk], source_lang)

    def translate_image(self, blk_list, image, inpainted_img, source_lang, target_lang, extra_context):
        translator = Translator(source_lang=source_lang, target_lang=target_lang, translator_key="Gemini", api_key="AIzaSyA8w-_E1d6tmfEnnGj6B6mq5ql1zmoCCFA")
        
        translator.translate(blk_list, image, extra_context)
        # unwanted_strings = ['COLAMANGA', '.com', 'acloudmerge.com', 'acloudmerge', 'COLAMANGA.com', 'newtoki464.com', 'newtoki464' , "가장 '업문세공사이트 웹툰왕국뉴토끼464 GHIIPS:LINEWTOKIASA.CON", "'가장' '입문재층사이트 웹툰왕국뉴토끼464. HHTTPS: JINEWTOKI4&4.CONI", "가장 '업문세공사이트 웹툰왕국뉴토끼464 GHIIPS:LINEWTOKIASA.CON"]
        # filtered_blk_list = filter_unwanted_text(blk_list, unwanted_strings)
        target_lang_en = get_language_code(target_lang)
        format_translations(blk_list, target_lang_en, upper_case=True)
        get_best_render_area(blk_list, image, inpainted_img)
        return blk_list

    def process_image(self, image_path, source_lang, target_lang, extra_context, show_masks=False):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading {image_path}")
            return

        # Create necessary directories
        output_dir = "output"
        cleaned_images_dir = output_dir + "/cleanedImages"
        translations_dir = output_dir + "/translations"
        for directory in [output_dir, cleaned_images_dir, translations_dir]:
            os.makedirs(directory, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_translated.png")
        cleaned_image_path = os.path.join(cleaned_images_dir, f"{base_name}_cleaned.png")
        translation_path = os.path.join(translations_dir, f"{base_name}_translation.json")

        blk_list = self.detect_blocks(image)

        if not blk_list:
            print(f"No text blocks found in {image_path}. Saving original image.")
            cv2.imwrite(output_path, image)
            return

        print("Step 1 - Text blocks detected.")
        self.OCR_image(image, blk_list, source_lang)
        mask = generate_mask(image, blk_list)

        # Get bubble boxes from detector
        bubble_boxes = self.block_detector_cache.get_last_bubble_boxes()

        print("Bubble Boxes:")
        for box, class_idx in bubble_boxes:
            print(f"Box: {box}, Class: {self.block_detector_cache.bubble_class_names[class_idx]}")

        # Generate visualization mask with both text blocks and bubbles
        visualization_mask = self.block_detector_cache.generate_visualization_mask(image, blk_list, bubble_boxes)

        if show_masks:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot the visualization mask
            ax1.imshow(cv2.cvtColor(visualization_mask, cv2.COLOR_BGR2RGB))
            ax1.set_title('Detected Text and Bubbles')
            ax1.axis('off')
            
            # Plot the inpainting mask
            ax2.imshow(mask, cmap='gray')  # Use grayscale colormap for the binary mask
            ax2.set_title('Inpainting Mask')
            ax2.axis('off')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('mask_output.png', bbox_inches='tight', dpi=300)
            plt.close()

            # Open the image with default photo viewer
            os.startfile('mask_output.png')

        cleaned_image = self.manual_inpaint(image, mask)
        cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
        print("Step 2 - Inpainting complete.")
        
        # # Save the cleaned image
        # cv2.imwrite(cleaned_image_path, cleaned_image)
        # print(f"Saved cleaned image to: {cleaned_image_path}")

        
        translated_blk_list = self.translate_image(blk_list, image, cleaned_image, source_lang, target_lang, extra_context)
        print("Step 3 - Translation complete.")

        # Prepare translation data for JSON
        # translation_data = {
        #     "image_name": base_name,
        #     "source_language": source_lang,
        #     "target_language": target_lang,
        #     "blocks": []
        # }

        # for blk in translated_blk_list:
        #     block_data = {
        #         "text": blk.text,
        #         "translation": blk.translation,
        #         "coordinates": blk.xyxy.tolist() if hasattr(blk.xyxy, 'tolist') else list(blk.xyxy),
        #         "bubble_coordinates": blk.bubble_xyxy.tolist() if hasattr(blk.bubble_xyxy, 'tolist') else list(blk.bubble_xyxy),
        #         "type" :blk.text_type,
        #         "font_color": blk.font_color,
        #         "outline_color": blk.outline_color,
        #         "bg_color": blk.bg_color,
        #         "alignment": blk.alignment if hasattr(blk, 'alignment') else None
        #     }
        #     translation_data["blocks"].append(block_data)

        # Save translation data to JSON
        # with open(translation_path, 'w', encoding='utf-8') as f:
        #     json.dump(translation_data, f, ensure_ascii=False, indent=2)
        # print(f"Saved translation data to: {translation_path}")

        set_alignment(translated_blk_list)
        rendered_image = draw_text(cleaned_image, translated_blk_list, bubble_boxes, font_pth='fonts/default.ttf', colour=(0, 0, 0))
        cv2.imwrite(output_path, rendered_image)
        print(f"Processed and saved final image: {output_path}")

    def process_image_with_retries(self, image_path, source_lang, target_lang, retries=3, delay=2, extra_context="", test_mode=True, test_count=3, current_count=0, show_masks=False):
        # Check if we've reached the test limit
        if test_mode and current_count >= test_count:
            print(f"\nTest mode: Reached limit of {test_count} images. Waiting for user input...")
            input("Press Enter to continue processing...")
            
        last_error = None
        for attempt in range(retries):
            try:
                self.process_image(image_path, source_lang, target_lang, extra_context, show_masks)
                return True
            except Exception as e:
                last_error = str(e)
                if attempt < retries - 1:  # Don't sleep on the last attempt
                    time.sleep(delay)
        
        # If we get here, all retries failed
        print(f"\nFailed to process {os.path.basename(image_path)} after {retries} attempts.")
        print(f"Last error: {last_error}")
        return False

    def batch_process(self, images_folder, source_lang, target_lang, extra_context, max_workers=1, show_masks=False):
        start_time = time.time()
        
        # Determine optimal number of workers based on available GPU memory
        if self.device == 'cuda':
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # Estimate memory per worker (adjust these values based on your model's requirements)
            estimated_memory_per_worker = 2 * 1024 * 1024 * 1024  # 2GB per worker
            max_possible_workers = max(1, int(total_memory / estimated_memory_per_worker))
            max_workers = min(max_workers, max_possible_workers)
            print(f"Adjusted number of workers to {max_workers} based on available GPU memory")
        
        # Initialize the inpainter pool
        self.initialize_inpainter_pool(max_workers)
        
        self.image_files = sorted([
            os.path.join(images_folder, f) 
            for f in os.listdir(images_folder) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not self.image_files:
            print("No images found in the specified folder.")
            return
        
        total_images = len(self.image_files)
        processed_images = 0
        failed_images = []
        
        print(f"\nStarting batch processing for {total_images} images using {max_workers} workers...")
        
        # Create progress bar
        pbar = tqdm(total=total_images, desc="Processing images")
        
        # Use ThreadPoolExecutor since we're managing the resources manually
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    self.process_image_with_retries, 
                    image_path, 
                    source_lang, 
                    target_lang, 
                    3,  # retries 
                    2,  # delay
                    extra_context,
                    True,  # test_mode
                    3,    # test_count
                    0,    # current_count
                    show_masks  # Add show_masks parameter
                ): image_path 
                for image_path in self.image_files
            }
            
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    success = future.result()
                    if success:
                        processed_images += 1
                    else:
                        failed_images.append(image_path)
                except Exception as e:
                    print(f"\nError processing {image_path}: {str(e)}")
                    failed_images.append(image_path)
                finally:
                    pbar.update(1)
        
        pbar.close()
        
        # Clean up inpainter pool
        self.inpainter_pool = None
        self.free_inpainters = Queue()

        # Process failed images sequentially
        if failed_images:
            print("\nRetrying failed images sequentially...")
            # Reinitialize a single inpainter for sequential processing
            self.initialize_inpainter_pool(1)
            
            for image_path in tqdm(failed_images, desc="Retrying failed"):
                try:
                    success = self.process_image_with_retries(
                        image_path, 
                        source_lang, 
                        target_lang,
                        retries=2,
                        delay=1,
                        extra_context=extra_context
                    )
                    if success:
                        processed_images += 1
                except Exception as e:
                    print(f"\nFinal attempt failed for {image_path}: {str(e)}")
            
            # Clean up the single inpainter
            self.inpainter_pool = None
            self.free_inpainters = Queue()

        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        print(f"\nBatch processing complete!")
        print(f"Time taken: {minutes} minutes and {seconds} seconds")
        print(f"Successfully processed: {processed_images}/{total_images} images")
        if processed_images < total_images:
            print(f"Failed to process: {total_images - processed_images} images")

        # Add memory cleanup at the end
        self.clear_gpu_memory()

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU available, using CPU")

if __name__ == "__main__":
    check_gpu()
    print("Hello World!")
