import cv2
import numpy as np
from scipy import ndimage

def detect_scratches(image, threshold=30):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Laplacian filter to detect edges
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Convert back to uint8
    laplacian = np.uint8(np.absolute(laplacian))

    # Threshold to get binary image
    _, binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def remove_scratches(image, mask):
    # Dilate the mask to cover the scratch completely
    kernel = np.ones((5,5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # Inpaint
    restored = cv2.inpaint(image, dilated_mask, 3, cv2.INPAINT_TELEA)

    return restored

def enhance_image(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the original A and B channels
    limg = cv2.merge((cl,a,b))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced

def restore_photo(input_path, output_path):
    # Read the image
    img = cv2.imread(input_path)

    # Detect scratches
    scratch_mask = detect_scratches(img)

    # Remove scratches
    restored = remove_scratches(img, scratch_mask)

    # Enhance the image
    enhanced = enhance_image(restored)

    # Save the result
    cv2.imwrite(output_path, enhanced)

    print(f"Restored image saved as {output_path}")

# Usage
restore_photo('old_photo.jpeg', 'restored_photo.jpg')