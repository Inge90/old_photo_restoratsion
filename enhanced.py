import cv2
import numpy as np

# Read the image
img = cv2.imread('old_photo2.jpg')

# Denoise the image
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# Sharpen the image
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(denoised, -1, kernel)

# Adjust contrast and brightness
alpha = 1.5 # Contrast control
beta = 30 # Brightness control
adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

# Save the result
cv2.imwrite('enhanced_photo.jpg', adjusted)