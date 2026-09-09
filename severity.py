from PIL import Image
import numpy as np

# SEVERITY THRESHOLDS
LOW_MAX = 1.0
MEDIUM_MAX = 5.0
HIGH_MAX = 10.0


def get_severity(area_percent):
    """
    Determine severity based on defect area percentage.
    """
    if area_percent < LOW_MAX:
        return "LOW", "🟢"

    elif area_percent < MEDIUM_MAX:
        return "MEDIUM", "🟡"

    elif area_percent < HIGH_MAX:
        return "HIGH", "🟠"

    else:
        return "CRITICAL", "🔴"


def severity_rank(severity):
    ranks = {
        "NONE": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4
    }
    return ranks.get(severity, 0)


def resize_mask(mask, width, height):
    if mask.shape == (height, width):
        return mask.astype(bool)

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)

    return np.array(mask_image) > 127