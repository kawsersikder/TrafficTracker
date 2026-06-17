import cv2
import numpy as np

def create_color_masks(image_path: str):
    """
    Reads an image and returns a dictionary of masks for different traffic colors.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Note: These HSV ranges might need fine-tuning for Google Maps
    # Green traffic
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Orange/Yellow traffic (moderate)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Red (heavy) vs Dark-red (severe/maroon) are split by BRIGHTNESS (V), not
    # just hue. Google's "red" is bright (V≳165); its "severe" maroon is darker
    # (V≈70–165) but still well above near-black, so the cutoff must sit there —
    # otherwise the maroon leaks into the plain-red bucket and dark_red ≈ 0.
    RED_V_MIN = 165

    # Red traffic (slow): bright reds only.
    lower_red1 = np.array([0, 90, RED_V_MIN])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 90, RED_V_MIN])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Dark Red traffic (severe): saturated reds dimmer than bright red.
    lower_dark_red1 = np.array([0, 90, 60])
    upper_dark_red1 = np.array([12, 255, RED_V_MIN - 1])
    lower_dark_red2 = np.array([158, 90, 60])
    upper_dark_red2 = np.array([179, 255, RED_V_MIN - 1])
    mask_dark_red = cv2.bitwise_or(cv2.inRange(hsv, lower_dark_red1, upper_dark_red1),
                                   cv2.inRange(hsv, lower_dark_red2, upper_dark_red2))

    return {
        "green": mask_green,
        "yellow": mask_yellow,
        "red": mask_red,
        "dark_red": mask_dark_red
    }
