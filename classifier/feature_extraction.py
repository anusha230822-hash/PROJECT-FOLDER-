"""
Feature Extraction Module for Satellite Images.

Extracts deterministic, explainable numerical features:
1. Color Features:
   - Mean & Standard Deviation for Red, Green, Blue channels
   - Mean Hue, Saturation, Value (HSV space)
2. Texture Features:
   - Grayscale Mean & Standard Deviation
   - Local contrast approximation
3. Structural Features:
   - Canny Edge Density
   - Perceived Brightness
   - Overall Pixel Variance
"""

import cv2
import numpy as np


def extract_color_features(img_rgb):
    """Calculates channel means, standard deviations, and HSV metrics."""
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    std_r = float(np.std(r))
    std_g = float(np.std(g))
    std_b = float(np.std(b))

    # Convert to HSV color space
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_mean = float(np.mean(img_hsv[:, :, 0]))
    s_mean = float(np.mean(img_hsv[:, :, 1]))
    v_mean = float(np.mean(img_hsv[:, :, 2]))

    return {
        'mean_r': round(mean_r, 4),
        'mean_g': round(mean_g, 4),
        'mean_b': round(mean_b, 4),
        'std_r': round(std_r, 4),
        'std_g': round(std_g, 4),
        'std_b': round(std_b, 4),
        'mean_hue': round(h_mean, 4),
        'mean_saturation': round(s_mean, 4),
        'mean_value': round(v_mean, 4),
    }


def extract_texture_features(img_rgb):
    """Calculates grayscale texture and contrast measurements."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))

    # Contrast approximation: RMS contrast (standard deviation of pixel intensities)
    contrast = float(gray_std)

    return {
        'gray_mean': round(gray_mean, 4),
        'gray_std': round(gray_std, 4),
        'contrast': round(contrast, 4),
    }


def extract_structural_features(img_rgb):
    """Calculates edge density and brightness metrics."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    total_pixels = gray.shape[0] * gray.shape[1]

    # Canny edge detector
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_count = int(np.count_nonzero(edges))
    edge_density = float(edge_count / total_pixels)

    # Brightness (perceived luminance)
    # 0.299*R + 0.587*G + 0.114*B standard formula
    brightness = float(np.mean(
        0.299 * img_rgb[:, :, 0] +
        0.587 * img_rgb[:, :, 1] +
        0.114 * img_rgb[:, :, 2]
    ))

    variance = float(np.var(gray))

    return {
        'edge_density': round(edge_density, 4),
        'edge_count': edge_count,
        'brightness': round(brightness, 4),
        'variance': round(variance, 4),
    }


FEATURE_NAMES = [
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b',
    'mean_hue', 'mean_saturation', 'mean_value',
    'gray_mean', 'gray_std', 'contrast',
    'edge_density', 'brightness', 'variance'
]


def extract_features(img_rgb):
    """
    Extracts all numerical features and returns:
    1. feature_vector (numpy 1D array of floats)
    2. feature_dict (key-value dictionary for UI display and JSON storage)
    """
    color = extract_color_features(img_rgb)
    texture = extract_texture_features(img_rgb)
    structural = extract_structural_features(img_rgb)

    feature_dict = {**color, **texture, **structural}

    vector = [
        feature_dict['mean_r'],
        feature_dict['mean_g'],
        feature_dict['mean_b'],
        feature_dict['std_r'],
        feature_dict['std_g'],
        feature_dict['std_b'],
        feature_dict['mean_hue'],
        feature_dict['mean_saturation'],
        feature_dict['mean_value'],
        feature_dict['gray_mean'],
        feature_dict['gray_std'],
        feature_dict['contrast'],
        feature_dict['edge_density'],
        feature_dict['brightness'],
        feature_dict['variance'],
    ]

    return np.array(vector, dtype=np.float32), feature_dict
