import numpy as np
from typing import List, Dict, Optional
from functools import lru_cache
import time
from .utils.textblock import TextBlock
from .rendering.render import cv2_to_pil
from .utils.translator_utils import encode_image_array, get_raw_text, set_texts_from_json, get_llm_client
from .utils.pipeline_utils import get_language_code
from deep_translator import GoogleTranslator
import google.generativeai as genai
import tenacity
import re
import logging
from datetime import datetime, timedelta
import threading
from collections import deque

class RateLimiter:
    def __init__(self, rpm_limit=15, rpd_limit=1500):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.requests = deque()
        self.daily_requests = deque()
        self.lock = threading.Lock()
        self.last_request_time = None
        self.min_delay = 4  # Minimum 4 seconds between requests
        
    def _cleanup_old_requests(self, queue, threshold):
        now = datetime.now()
        while queue and queue[0] < threshold:
            queue.popleft()
    
    def _enforce_minimum_delay(self):
        if self.last_request_time:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            
    def can_make_request(self) -> bool:
        with self.lock:
            now = datetime.now()
            
            # Cleanup old requests
            self._cleanup_old_requests(self.requests, now - timedelta(minutes=1))
            self._cleanup_old_requests(self.daily_requests, now - timedelta(days=1))
            
            # Check limits
            if len(self.requests) >= self.rpm_limit:
                return False
            if len(self.daily_requests) >= self.rpd_limit:
                return False
            
            self._enforce_minimum_delay()
            
            # Add new request timestamp
            self.requests.append(now)
            self.daily_requests.append(now)
            self.last_request_time = now
            return True
            
    def wait_if_needed(self):
        attempts = 0
        max_attempts = 5  # Maximum number of attempts before giving up
        
        while attempts < max_attempts:
            if self.can_make_request():
                return True
            
            # Calculate wait time based on current state
            if len(self.requests) >= self.rpm_limit:
                # Wait until the oldest request expires from the minute window
                wait_time = 61 - (datetime.now() - self.requests[0]).total_seconds()
            else:
                wait_time = self.min_delay
            
            time.sleep(min(wait_time, 10))  # Cap maximum wait time at 10 seconds
            attempts += 1
            
        return False  # Return False if we couldn't make request after max attempts

class Translator:
    def __init__(self, source_lang: str = "", target_lang: str = "", translator_key: str = "Google Translate", api_key: str = None):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator_key = translator_key
        self.api_key = api_key
        self.client = get_llm_client(self.translator_key, self.api_key)
        self.translation_cache: Dict[str, str] = {}
        self.batch_size = 10  # Number of text blocks to translate at once
        self.min_text_length = 1  # Minimum characters to translate
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = RateLimiter() if 'Gemini' in translator_key else None
        
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=10),
        retry=tenacity.retry_if_exception_type((Exception))
    )
    def get_gemini_translation(self, user_prompt: str, system_prompt: str, image) -> str:
        try:
            # Add rate limiting with timeout
            if self.rate_limiter:
                if not self.rate_limiter.wait_if_needed():
                    raise Exception("Rate limit reached and maximum wait time exceeded")
                
            generation_config = {
                "temperature": 0.5,
                "top_p": 1,
                "top_k": 40,
            }
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            model_instance = self.client.GenerativeModel(
                model_name = "gemini-1.5-flash-latest",
                generation_config=generation_config,
                system_instruction=system_prompt,
                safety_settings=safety_settings
            )
            
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model_instance.generate_content([image, combined_prompt])
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
                
            return response.text
            
        except Exception as e:
            print(f"Error in Gemini translation: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def get_cached_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Cache translations to avoid repeated API calls"""
        cache_key = f"{text}:{source_lang}:{target_lang}"
        return self.translation_cache.get(cache_key)

    def is_valid_text(self, text: str) -> bool:
        """Check if text is valid for translation"""
        # Remove whitespace and check if there's actual content
        cleaned_text = text.strip()
        
        # Check for common patterns that should be translated
        patterns = [
            r'[.]{2,}',  # Ellipsis
            r'[!?]+',    # Exclamation or question marks
            r'[가-힣]+',  # Korean characters
            r'[一-龯]+',  # Chinese characters
            r'[ぁ-んァ-ン]+',  # Japanese characters
        ]
        
        if any(re.search(pattern, cleaned_text) for pattern in patterns):
            return True
            
        return bool(cleaned_text and len(cleaned_text) >= self.min_text_length)

    def validate_translation(self, original: str, translation: str) -> bool:
        """Enhanced validation of translation"""
        if not translation:
            self.logger.warning(f"Empty translation for: '{original}'")
            return False
            
        # Special handling for very short text (like "저건...!")
        if len(original) <= 5:
            # Allow shorter translations for short original text
            if len(translation) < 1:
                self.logger.warning(f"Translation too short for: '{original}' -> '{translation}'")
                return False
            return True
            
        # Regular validation for longer text
        if len(translation) < len(original) * 0.3:
            self.logger.warning(f"Translation too short: '{original}' -> '{translation}'")
            return False
        if len(translation) > len(original) * 3:
            self.logger.warning(f"Translation too long: '{original}' -> '{translation}'")
            return False
            
        return True

    def translate_batch(self, blocks: List[TextBlock], source_lang_code: str, target_lang_code: str):
        """Translate a batch of text blocks with improved handling"""
        # Filter out invalid texts and log them
        valid_blocks = []
        for blk in blocks:
            if self.is_valid_text(blk.text):
                valid_blocks.append(blk)
            else:
                self.logger.warning(f"Skipping invalid text in batch: '{blk.text}'")
                blk.translation = blk.text

        if not valid_blocks:
            return

        texts = [blk.text for blk in valid_blocks]
        combined_text = "\n".join(texts)
        
        try:
            if self.translator_key == "Google Translate":
                translator = GoogleTranslator(source=source_lang_code, target=target_lang_code)
                translation = translator.translate(combined_text)
                translations = translation.split("\n")
                
                for blk, trans in zip(valid_blocks, translations):
                    if self.validate_translation(blk.text, trans):
                        blk.translation = trans
                        print(f"\n=== Batch Translation Result ===")
                        print(f"Original ({source_lang_code}): {blk.text}")
                        print(f"Translation ({target_lang_code}): {trans}")
                        print("============================\n")
                    else:
                        # Fallback to individual translation for failed cases
                        self.logger.warning(f"Batch translation failed, trying individual for: '{blk.text}'")
                        self.translate_single(blk, source_lang_code, target_lang_code)
                    
        except Exception as e:
            self.logger.error(f"Batch translation error: {str(e)}")
            # Fallback to individual translation
            for blk in valid_blocks:
                self.translate_single(blk, source_lang_code, target_lang_code)

    def translate_single(self, blk: TextBlock, source_lang_code: str, target_lang_code: str):
        """Translate a single text block with improved handling"""
        try:
            if not self.is_valid_text(blk.text):
                self.logger.warning(f"Skipping invalid text: '{blk.text}'")
                blk.translation = blk.text
                return

            # Check cache first
            cached = self.get_cached_translation(blk.text, source_lang_code, target_lang_code)
            if cached:
                self.logger.info(f"Cache hit for: '{blk.text}' -> '{cached}'")
                blk.translation = cached
                return

            text = blk.text.replace(" ", "") if 'zh' in source_lang_code.lower() or source_lang_code.lower() == 'ja' else blk.text
            
            # Try primary translator
            translator = GoogleTranslator(source='auto', target=target_lang_code)
            translation = translator.translate(text)
            
            self.logger.info(f"Translating: '{text}' -> '{translation}'")
            
            if self.validate_translation(text, translation):
                blk.translation = translation
                print(f"\n=== Translation Result ===")
                print(f"Original ({source_lang_code}): {text}")
                print(f"Translation ({target_lang_code}): {translation}")
                print("=======================\n")
                self.translation_cache[f"{text}:{source_lang_code}:{target_lang_code}"] = translation
            else:
                # Fallback to alternative translation method
                self.logger.warning(f"Primary translation failed, trying fallback for: '{text}'")
                fallback_translator = GoogleTranslator(source=source_lang_code, target=target_lang_code)
                translation = fallback_translator.translate(text)
                if self.validate_translation(text, translation):
                    blk.translation = translation
                else:
                    self.logger.error(f"Both primary and fallback translation failed for: '{text}'")
                    blk.translation = text
                    
        except Exception as e:
            self.logger.error(f"Translation error for text '{text}': {str(e)}")
            blk.translation = text

    def translate(self, blk_list: List[TextBlock], image: np.ndarray, extra_context: str) -> List[TextBlock]:
        source_lang_code = get_language_code(self.source_lang)
        target_lang_code = get_language_code(self.target_lang)

        if self.translator_key == "Google Translate":
            # Process in batches
            for i in range(0, len(blk_list), self.batch_size):
                batch = blk_list[i:i + self.batch_size]
                self.translate_batch(batch, source_lang_code, target_lang_code)

        elif 'Gemini' in self.translator_key:
            try:
                # Split into smaller batches for Gemini to avoid token limits
                batch_size = 3  # Reduced batch size to minimize token usage
                for i in range(0, len(blk_list), batch_size):
                    batch = blk_list[i:i + batch_size]
                    
                    try:
                        system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
                        batch_raw_text = get_raw_text(batch)
                        user_prompt = f"{extra_context}\nTranslate this:\n{batch_raw_text}"

                        image_pil = cv2_to_pil(image)
                        batch_translated_text = self.get_gemini_translation(user_prompt, system_prompt, image_pil)
                        
                        if batch_translated_text:
                            print(f"\n=== Gemini Translation Result (Batch {i//batch_size + 1}) ===")
                            print(f"Original Text:\n{batch_raw_text}")
                            print(f"\nTranslated Text:\n{batch_translated_text}")
                            print("============================\n")
                            set_texts_from_json(batch, batch_translated_text)
                            
                            # Add a small delay between successful batches
                            time.sleep(2)
                            
                    except Exception as batch_error:
                        print(f"Error in batch {i//batch_size + 1}: {str(batch_error)}")
                        # Fallback to Google Translate for this specific batch
                        print(f"Falling back to Google Translate for batch {i//batch_size + 1}...")
                        self.translate_batch(batch, source_lang_code, target_lang_code)
                        
            except Exception as e:
                print(f"Gemini translation error: {str(e)}")
                # Fallback to Google Translate for all remaining text
                self.translator_key = "Google Translate"
                self.translate(blk_list, image, extra_context)
                self.translator_key = "Gemini AI"

        return blk_list

    def get_system_prompt(self, source_lang: str, target_lang: str):
        return f"""
        Translate from {source_lang} to {target_lang} with natural, fluent language that fits the comic’s tone and context. Follow these rules:

        1. **Natural Flow**: Ensure translations sound natural to native {target_lang} speakers.
        2. **No Direct Pronouns**: Avoid pronouns like "당신" (Korean) or "彼" (Japanese) unless necessary.
        3. **Fix OCR Errors**: Correct obvious typos based on context.
        4. **Remove Watermarks/Links**: Ignore URLs or watermarks; return an empty string for them.
        5. **Use Visual Cues**: Refer to the comic’s images for emotions and scene context.
        6. **Match Style**: Keep casual, formal, or slang styles as appropriate for the scene.
        7. **Fantasy Terms**: Use consistent translations for fantasy names and titles.
        8. **Translate Only Values**: In JSON, translate only the values, not keys.
        9. **Onomatopoeia**: Adapt sound effects to suit {target_lang}.

        **IMPORTANT**:
        - **No Notes/Comments**.
        - **Respect Space**: Keep translations concise to fit speech bubbles.
        - **Use Only {target_lang}**: Return translations only in {target_lang}.
        """
