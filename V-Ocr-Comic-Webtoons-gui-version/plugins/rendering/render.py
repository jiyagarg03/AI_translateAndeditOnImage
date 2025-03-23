import cv2
import numpy as np
from typing import Tuple, List, Union

from PIL import Image, ImageFont, ImageDraw, ImageOps
from PySide6.QtGui import QFontMetrics, QFont

from .hyphen_textwrap import wrap as hyphen_wrap
from ..utils.textblock import TextBlock
from ..detection import make_bubble_mask, bubble_interior_bounds
from ..utils.textblock import adjust_blks_size

def cv2_to_pil(cv2_image: np.ndarray):
    # Convert color channels from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def pil_to_cv2(pil_image: Image):
    # Convert the PIL image to a numpy array
    numpy_image = np.array(pil_image)
    
    # PIL images are in RGB by default, OpenCV uses BGR, so convert the color space
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return cv2_image

from PIL import ImageDraw, ImageFont
import textwrap
import numpy as np

def pil_word_wrap(image: Image, tbbox_top_left: Tuple, font_pth: str, text: str, 
                  roi_width, roi_height, align: str, spacing, init_font_size: int, min_font_size: int = 10):
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""
    mutable_message = text
    font_size = init_font_size
    font = ImageFont.truetype(font_pth, font_size)

    def eval_metrics(txt, font):
        """Quick helper function to calculate width/height of text."""
        (left, top, right, bottom) = ImageDraw.Draw(image).multiline_textbbox(xy=tbbox_top_left, text=txt, font=font, align=align, spacing=spacing)
        return (right-left, bottom-top)

    while font_size > min_font_size:
        font = font.font_variant(size=font_size)
        width, height = eval_metrics(mutable_message, font)
        if height > roi_height:
            font_size -= 0.75  # Reduce pointsize
            mutable_message = text  # Restore original text
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                columns -= 1
                if columns == 0:
                    break
                mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
                wrapped_width, _ = eval_metrics(mutable_message, font)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                font_size -= 0.75  # Reduce pointsize
                mutable_message = text  # Restore original text
        else:
            break

    if font_size <= min_font_size:
        font_size = min_font_size
        mutable_message = text
        font = font.font_variant(size=font_size)

        # Wrap text to fit within as much as possible
        # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
        # This is a brute force approach, but it works well enough
        min_cost = 1e9
        min_text = text
        for columns in range(1, len(text)):
            wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True))
            wrapped_width, wrapped_height = eval_metrics(wrapped_text, font)
            cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
            if cost < min_cost:
                min_cost = cost
                min_text = wrapped_text

        mutable_message = min_text

    return mutable_message, font_size

def get_text_size(font_path: str, font_size: int, text: str) -> Tuple[int, int]:
    """Get the pixel dimensions of text with given font and size."""
    font = ImageFont.truetype(font_path, font_size)
    # Use getbbox() instead of deprecated getsize()
    bbox = font.getbbox(text)
    # bbox returns (left, top, right, bottom)
    # Convert to (width, height)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])

def get_gradient_color(start_color: Tuple[int, int, int], 
                      end_color: Tuple[int, int, int], 
                      percent: float) -> Tuple[int, int, int]:
    """Calculate color at given percentage between start and end colors"""
    r = int(start_color[0] + (end_color[0] - start_color[0]) * percent)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * percent)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * percent)
    return (r, g, b)

def create_gradient_mask(width: int, height: int, start_color: Tuple[int, int, int], 
                        end_color: Tuple[int, int, int], vertical: bool = True) -> Image:
    """Create a gradient mask image"""
    base = Image.new('RGB', (width, height), start_color)
    top = Image.new('RGB', (width, height), end_color)
    mask = Image.new('L', (width, height))
    mask_data = []
    
    for y in range(height):
        for x in range(width):
            if vertical:
                # Vertical gradient
                mask_data.append(int(255 * (y / height)))
            else:
                # Horizontal gradient
                mask_data.append(int(255 * (x / width)))
    
    mask.putdata(mask_data)
    return Image.composite(top, base, mask)

def write_text_box(image: Image, position: Tuple[int, int], text: str, box_width: int, 
                  font_path: str, box_height: int, font_size: int = 11, 
                  color: Union[Tuple[int, int, int], Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = (0, 0, 0),
                  outline_thickness: int = 0,
                  outline_color: Tuple[int, int, int] = None,
                  alignment: str = "left") -> Tuple[int, int]:
    """
    Write text in a box with gradient effect.
    """
    x, y = position
    draw = ImageDraw.Draw(image)
    
    # Start with a very large font size
    current_font_size = box_height  # Start with height of box as font size
    best_font_size = None
    best_lines = None
    
    def text_fits(lines, font_size):
        """Check if text configuration fits within box bounds"""
        font = ImageFont.truetype(font_path, font_size)
        
        # Check height
        bbox = font.getbbox('Aj')
        line_height = (bbox[3] - bbox[1]) * 1.2
        total_height = line_height * len(lines)
        if total_height > box_height:
            return False
            
        # Check width of each line
        for line in lines:
            size = get_text_size(font_path, font_size, line)
            if size[0] > box_width:
                return False
                
        return True

    # Binary search for largest fitting font size
    min_size = 10
    max_size = current_font_size
    
    while min_size <= max_size:
        current_font_size = (min_size + max_size) // 2
        font = ImageFont.truetype(font_path, current_font_size)
        
        # Try single line first
        if text_fits([text], current_font_size):
            best_font_size = current_font_size
            best_lines = [text]
            min_size = current_font_size + 1
            continue
            
        # If single line doesn't fit, try wrapping
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            size = get_text_size(font_path, current_font_size, test_line)
            
            if size[0] <= box_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
            
        if text_fits(lines, current_font_size):
            best_font_size = current_font_size
            best_lines = lines
            min_size = current_font_size + 1
        else:
            max_size = current_font_size - 1

    if best_font_size is None or best_lines is None:
        best_font_size = min_size
        best_lines = [text]

    # Use the best result found
    font = ImageFont.truetype(font_path, best_font_size)
    bbox = font.getbbox('Aj')
    line_height = (bbox[3] - bbox[1]) * 1.2
    total_height = line_height * len(best_lines)
    
    # Center text vertically in box
    start_y = y + (box_height - total_height) / 2

    # Draw outline if specified
    if outline_thickness > 0 and outline_color:
        for line_idx, line in enumerate(best_lines):
            size = get_text_size(font_path, best_font_size, line)
            if alignment == "center":
                line_x = x + (box_width - size[0]) / 2
            elif alignment == "justify" and line_idx < len(best_lines) - 1:
                words = line.split()
                line_without_spaces = ''.join(words)
                total_size = get_text_size(font_path, best_font_size, line_without_spaces)
                space_width = (box_width - total_size[0]) / (len(words) - 1.0)
                start_x = x + (box_width - total_size[0]) / 2  # Center justify
                for word in words[:-1]:
                    draw.text((start_x, start_y + line_idx * line_height), word, font=font, fill=outline_color)
                    word_size = get_text_size(font_path, best_font_size, word)
                    start_x += word_size[0] + space_width
                last_word_size = get_text_size(font_path, best_font_size, words[-1])
                last_word_x = x + box_width - last_word_size[0]
                draw.text((last_word_x, start_y + line_idx * line_height), words[-1], font=font, fill=outline_color)
                continue
            else:
                line_x = x
            line_y = start_y + line_idx * line_height
            
            for dx, dy in [(dx, dy) for dx in range(-outline_thickness, outline_thickness + 1)
                          for dy in range(-outline_thickness, outline_thickness + 1)
                          if dx != 0 or dy != 0]:
                draw.text((line_x + dx, line_y + dy), line, 
                         font=font, fill=outline_color)

    # Create a temporary image for the text
    is_gradient = isinstance(color, tuple) and isinstance(color[0], tuple)
    if is_gradient:
        start_color, end_color = color
        # Create temporary image with alpha channel
        text_overlay = Image.new('RGBA', (box_width, box_height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_overlay)
        
        # Draw text in white first
        for line_idx, line in enumerate(best_lines):
            size = get_text_size(font_path, best_font_size, line)
            if alignment == "center":
                line_x = (box_width - size[0]) / 2
            else:
                line_x = 0
            line_y = start_y + line_idx * line_height - y
            text_draw.text((line_x, line_y), line, font=font, fill=(255, 255, 255, 255))
        
        # Create gradient mask
        gradient = create_gradient_mask(box_width, box_height, start_color, end_color)
        
        # Apply gradient to text
        text_mask = text_overlay.convert('L')
        gradient_text = Image.composite(gradient, Image.new('RGB', (box_width, box_height), (0, 0, 0)), text_mask)
        
        # Paste the gradient text onto the main image
        image.paste(gradient_text, (x, y), text_mask)
    else:
        # Regular single color text
        for line_idx, line in enumerate(best_lines):
            size = get_text_size(font_path, best_font_size, line)
            if alignment == "center":
                line_x = x + (box_width - size[0]) / 2
            else:
                line_x = x
            line_y = start_y + line_idx * line_height
            draw.text((line_x, line_y), line, font=font, fill=color)

    return box_width, int(total_height)

def draw_text(image: np.ndarray, blk_list: List[TextBlock], bubble_boxes: List[Tuple[np.ndarray, int]], font_pth: str, 
              colour: str = "#000", init_font_size: int = 40, 
              min_font_size: int = 10, outline: bool = True):
    """
    Enhanced function to draw text onto the image with better text wrapping and positioning.
    Uses larger initial font sizes and better text fitting.
    """
    print("====== Drawing text... ======")
    image = cv2_to_pil(image)
    draw = ImageDraw.Draw(image)
    
    # Define bubble class names
    bubble_class_names = ['continuation', 'dialog', 'misc-text', 'narration', 
                          'only-text', 'small-text', 'sound-effect', 'thought']
    
    last_type = None  # Track the last non-continuation type
    
    for blk in blk_list:
        x1, y1, x2, y2 = blk.xyxy
        
        # Check if the text block is inside any bubble box
        for box, class_idx in bubble_boxes:
            bx1, by1, bx2, by2 = box
            if x1 >= bx1 and y1 >= by1 and x2 <= bx2 and y2 <= by2:
                blk.type = bubble_class_names[class_idx]
                if blk.type != "continuation":
                    last_type = blk.type  # Update last_type if not continuation
                else:
                    blk.type = last_type  # Use last_type for continuation
                print(f"Type ======> {blk.type} --- {blk.translation}")
                break
        
        translation = blk.translation
        colour = blk.font_color
        outline_color = blk.outline_color

        if not translation or len(translation) == 1:
            continue

        # Convert colors from hex to RGB if needed
        if isinstance(colour, str) and colour.startswith('#'):
            # Create gradient from red to black
            start_color = (255, 0, 0)  # Red
            end_color = (0, 0, 0)      # Black
            colour = (start_color, end_color)  # Tuple of gradient colors

        # Start with a larger initial font size
        font_size = init_font_size * 2  # Double the initial font size

        # Set font style based on blk.type
        if blk.type == "narration":
            alignment = "center"
            font_pth = 'fonts/BebasNeue-Regular.ttf'
        elif blk.type == "dialog":
            alignment = "center"
            font_pth = 'fonts/CCAskForMercy-Regular.otf'
        elif blk.type == "thought":
            alignment = "center"
            font_pth = 'fonts/CCAskForMercy-Italic.otf'
        else:
            alignment = "center"
            font_pth = 'fonts/CCAskForMercy-Regular.otf'

        # Draw text with outline if enabled
        outline_thickness = 2 if outline else 0
        write_text_box(image, (x1, y1), translation, x2 - x1, font_pth, y2 - y1,
                      font_size=font_size, color=colour,
                      outline_thickness=outline_thickness,
                      outline_color=outline_color,
                      alignment=alignment)

    # Convert back to OpenCV format
    return pil_to_cv2(image)


def get_best_render_area(blk_list: List[TextBlock], img, inpainted_img):
    # Using Speech Bubble detection to find best Text Render Area
    for blk in blk_list:
        if blk.text_class == 'text_bubble':
            bx1, by1, bx2, by2 = blk.bubble_xyxy
            bubble_clean_frame = inpainted_img[by1:by2, bx1:bx2]
            bubble_mask = make_bubble_mask(bubble_clean_frame)
            text_draw_bounds = bubble_interior_bounds(bubble_mask)

            bdx1, bdy1, bdx2, bdy2 = text_draw_bounds

            bdx1 += bx1
            bdy1 += by1

            bdx2 += bx1
            bdy2 += by1

            if blk.source_lang == 'ja':
                blk.xyxy[:] = [bdx1, bdy1, bdx2, bdy2]
                adjust_blks_size(blk_list, img, -5, -5)
            else:
                tx1, ty1, tx2, ty2  = blk.xyxy

                nx1 = max(bdx1, tx1)
                nx2 = min(bdx2, tx2)
                
                blk.xyxy[:] = [nx1, ty1, nx2, ty2]

    return blk_list


def pyside_word_wrap(text: str, font_input: str, roi_width: int, roi_height: int,
                    line_spacing, outline_width, bold, italic, underline,
                    init_font_size: int, min_font_size: int = 10) -> Tuple[str, int]:
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""
    
    def get_text_height(text, font, line_spacing):
        font_metrics = QFontMetrics(font)
        lines = text.split('\n')
        single_line_height = font_metrics.height()
        single_line_height += outline_width * 2
        total_height = single_line_height * len(lines)
        extra_spacing = single_line_height * (line_spacing - 1) * (len(lines) - 1)
        
        return total_height + extra_spacing
    
    def get_text_width(txt, font):
        fm = QFontMetrics(font)
        max_width = max(fm.horizontalAdvance(line) for line in txt.split('\n'))

        return max_width
    
    def prepare_font(font_size):
        font = QFont(font_input, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        font.setUnderline(underline)

        return font
    
    def eval_metrics(txt: str, font_sz) -> Tuple[float, float]:
        """Quick helper function to calculate width/height of text."""

        font = prepare_font(font_sz)
        width = get_text_width(txt, font)
        height = get_text_height(txt, font, line_spacing)

        return width, height

    mutable_message = text
    font_size = init_font_size
    
    while font_size > min_font_size:
        width, height = eval_metrics(mutable_message, font_size)
        if height > roi_height:
            font_size -= 1  # Reduce pointsize
            mutable_message = text  # Restore original text
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                columns -= 1
                if columns == 0:
                    break
                mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True)) 
                wrapped_width, _ = eval_metrics(mutable_message, font_size)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                font_size -= 1  # Reduce pointsize
                mutable_message = text  # Restore original text
        else:
            break

    if font_size <= min_font_size:
        font_size = min_font_size
        mutable_message = text

        # Wrap text to fit within as much as possible
        # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
        min_cost = 1e9
        min_text = text
        for columns in range(1, len(text)):
            wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=False, break_long_words=False, hyphenate_broken_words=True))
            wrapped_width, wrapped_height = eval_metrics(wrapped_text, font_size)
            cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
            if cost < min_cost:
                min_cost = cost
                min_text = wrapped_text

        mutable_message = min_text

    return mutable_message, font_size

def manual_wrap(main_page, blk_list: List[TextBlock], font_family: str, line_spacing, 
                outline_width, bold, italic, underline, init_font_size: int = 40, 
         min_font_size: int = 10):
    
    for blk in blk_list:
        x1, y1, width, height = blk.xywh

        translation = blk.translation
        if not translation or len(translation) == 1:
            continue

        translation, font_size = pyside_word_wrap(translation, font_family, width, height,
                                                 line_spacing, outline_width, bold, italic, underline,
                                                 init_font_size, min_font_size)
 
        main_page.blk_rendered.emit(translation, font_size, blk)



        
