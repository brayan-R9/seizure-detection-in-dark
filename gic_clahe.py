import cv2
import numpy as np

def apply_gic(image, gamma=0.5):
    """
    Apply Gamma Intense Correction (GIC) to enhance image visibility.

    Parameters:
        image (numpy.ndarray): Input image.
        gamma (float): Gamma correction factor. Use <1 to brighten, >1 to darken.

    Returns:
        numpy.ndarray: Enhanced image.
    """
    # Normalize pixel values to [0, 1]
    normalized = image / 255.0
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    # Scale back to [0, 255]
    enhanced = (corrected * 255).astype(np.uint8)
    return enhanced

def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return enhanced

def apply_gic_with_clahe(image, gamma=0.5):
    """
    Apply GIC followed by CLAHE to enhance the image.

    Parameters:
        image (numpy.ndarray): Input image.
        gamma (float): Gamma correction factor.

    Returns:
        numpy.ndarray: Enhanced image.
    """
    clahe_enhanced = apply_clahe(image)
    return apply_gic(clahe_enhanced, gamma)


