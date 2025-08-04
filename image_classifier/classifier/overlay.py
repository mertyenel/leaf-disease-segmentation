import cv2
import numpy as np

def create_overlay(original_path, mask_path, output_path=None):
    if output_path is None:
        output_path = original_path.replace('.', '_overlay.')

    overlay_color_bgr = [255, 40, 255]
    
    alpha = 0.2
    beta = 1 - alpha
    
    threshold_value = 20

    orig = cv2.imread(original_path)
    mask_bgr = cv2.imread(mask_path)

    mask_resized = cv2.resize(mask_bgr, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_red_channel = mask_resized[:, :, 2]
    mask_region = mask_red_channel > threshold_value

    result = orig.copy()

    overlay = np.zeros_like(orig, dtype=np.uint8)
    overlay[:] = overlay_color_bgr

    result[mask_region] = cv2.addWeighted(
        orig[mask_region], beta, overlay[mask_region], alpha, 0
    )

    cv2.imwrite(output_path, result)
    
    return output_path 