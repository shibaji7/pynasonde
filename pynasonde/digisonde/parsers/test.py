import cv2
import numpy as np
import pytesseract
from PIL import Image

# Path to the uploaded image
image_path = "../../ERAUCodeBase/APEP/tmp/ion_BC840_000005.png"

# Load the image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Crop the left side where the table exists
# Adjust x, y, w, h if needed based on image variations
x, y, w, h = 0, 30, 150, image.shape[0]
table_crop = gray[y : y + h, x : x + w]

# Optional: Apply thresholding to enhance text clarity
thresh = cv2.adaptiveThreshold(
    table_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# OCR configuration
custom_oem_psm_config = r"--oem 3 --psm 6"  # Assume a uniform block of text

# Run Tesseract OCR
extracted_text = pytesseract.image_to_string(thresh, config=custom_oem_psm_config)

# Print the raw OCR result
print("Raw OCR output:\n")
print(extracted_text)

# Parse the table into key-value pairs
table_data = {}
for line in extracted_text.splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        table_data[key.strip()] = value.strip()

# Display the parsed table
print("\nParsed Table:")
for k, v in table_data.items():
    print(f"{k}: {v}")
