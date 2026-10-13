"""
Image Preprocessing Module for Satellite Images.
Follows a clean, modular pipeline:
1. Load image (Pillow / OpenCV)
2. Convert to RGB color space
3. Resize to consistent dimensions (128x128)
4. Apply noise reduction (Gaussian blur)
5. Normalize pixel intensity
"""

import os
import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (128, 128)


def load_image(image_input):
    """
    Loads an image from file path, file object, or numpy array.
    Returns RGB uint8 numpy array.
    """
    if isinstance(image_input, (str, os.PathLike)):
        # Read using OpenCV
        img_bgr = cv2.imread(str(image_input))
        if img_bgr is None:
            # Fallback to Pillow
            pil_img = Image.open(image_input)
            img_rgb = np.array(pil_img.convert('RGB'))
            return img_rgb
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    elif hasattr(image_input, 'read'):
        # Uploaded file / BytesIO object
        image_input.seek(0)
        pil_img = Image.open(image_input)
        img_rgb = np.array(pil_img.convert('RGB'))
        return img_rgb
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        return image_input
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert('RGB'))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def resize_image(img_rgb, target_size=TARGET_SIZE):
    """Resizes RGB image to specified target size (width, height)."""
    return cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)


def denoise_image(img_rgb, kernel_size=(3, 3)):
    """Applies a gentle Gaussian blur to reduce high-frequency sensor noise."""
    return cv2.GaussianBlur(img_rgb, kernel_size, 0)


def enhance_image(img_rgb):
    """
    Applies contrast enhancement (CLAHE on luminance channel)
    to reveal ground-cover boundaries in satellite imagery.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def normalize_image(img_rgb):
    """Normalizes RGB uint8 [0, 255] values to floating point [0.0, 1.0]."""
    return img_rgb.astype(np.float32) / 255.0


def preprocess_image(image_input, target_size=TARGET_SIZE):
    """
    Full preprocessing pipeline:
    1. Load image and ensure RGB
    2. Resize to consistent 128x128
    3. Denoise with Gaussian blur
    4. Optional light enhancement
    Returns tuple: (processed_uint8_rgb, normalized_float_rgb)
    """
    img_rgb = load_image(image_input)
    resized = resize_image(img_rgb, target_size=target_size)
    denoised = denoise_image(resized)
    normalized = normalize_image(denoised)
    return denoised, normalized
