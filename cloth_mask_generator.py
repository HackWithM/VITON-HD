import cv2
import numpy as np
import os
from PIL import Image

def generate_cloth_mask(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cloth_filenames = os.listdir(input_dir)

    for fname in cloth_filenames:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        # Read cloth image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Cannot read {input_path}, skipping.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to create mask (tune threshold if needed)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Optional: Morphological ops to clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Save mask as png or jpg in output dir
        cv2.imwrite(output_path, mask)
        print(f"Mask saved: {output_path}")

if __name__ == "__main__":
    cloth_dir = "test_data/cloth"
    mask_dir = "test_data/cloth-mask"
    generate_cloth_mask(cloth_dir, mask_dir)
