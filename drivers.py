import easyocr
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import re

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(image):
    """Enhance image contrast, denoise, and apply adaptive thresholding."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Bilateral Filtering (Noise Reduction)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive Thresholding to enhance text
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

    # Morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

def extract_license_number(ocr_results):
    """Extracts License Number with any uppercase letter as prefix."""
    license_no = None
    text_lines = [text.upper() for _, text, _ in ocr_results]

    # License Number Patterns (Now allowing ANY letter at start)
    license_patterns = [
        r"[A-Z]\d{2}-\d{2}-\d{6}",  # A02-19-016056 / X19-23-987654
        r"\d{2}-\d{2}-\d{6}",       # 02-19-016056 (missing prefix)
        r"[A-Z]\d{2}\s\d{2}\s\d{6}" # A02 19 016056 (OCR misreads dashes as spaces)
    ]
    
    for line in text_lines:
        for pattern in license_patterns:
            match = re.search(pattern, line)
            if match:
                license_no = match.group(0).replace(" ", "-")  # Convert spaces to dashes
                
                # If missing a starting letter, assume generic "X" (or leave blank)
                if re.match(r"\d{2}-\d{2}-\d{6}", license_no):  
                    license_no = "A" + license_no  # Use 'X' as a generic placeholder

                return license_no  # Return as soon as a match is found

    return None

def extract_name(full_name):
    """
    Extracts last name, first name, and middle name correctly from 
    a full name formatted as 'LASTNAME, FIRSTNAME MIDDLENAME'.
    """
    last_name, first_name, middle_name = None, None, None

    if not full_name or "," not in full_name:
        return None, None, None  # Invalid format

    # Split last name from the rest
    parts = full_name.split(", ")
    if len(parts) != 2:
        return None, None, None  # Invalid format

    last_name = parts[0].strip()
    first_middle = parts[1].split()

    # Correctly handle different name cases
    if len(first_middle) == 1:
        first_name = first_middle[0]  # Only first name, no middle name
    elif len(first_middle) == 2:
        first_name, middle_name = first_middle  # First name and middle name correctly assigned
    elif len(first_middle) > 2:
        first_name = first_middle[0] + " " + first_middle[1]  # First two words = first name
        middle_name = " ".join(first_middle[2:])  # Remaining words = middle name

    return last_name, first_name, middle_name


def extract_info(ocr_results):
    """Extracts key details (License No., Full Name, Date of Birth)."""
    license_no, full_name, birthdate = None, None, None
    text_lines = [text.upper().strip() for _, text, _ in ocr_results]

    # Extract License Number
    license_no = extract_license_number(ocr_results)

    # Full Name Pattern (LASTNAME, FIRSTNAME MIDDLENAME)
    name_pattern = re.compile(r"([A-Z]+),\s([A-Z]+(?:\s[A-Z]+)*)(?:\s([A-Z]+(?:\s[A-Z]+)*))?")
    
    for line in text_lines:
        match = name_pattern.search(line)
        if match:
            full_name = match.group(0)
            break

    # Extract Date of Birth
    date_pattern = r"(\d{4}/\d{2}/\d{2})"
    for line in text_lines:
        match = re.search(date_pattern, line)
        if match:
            birthdate = match.group(0)
            break

    # Extract Names Properly
    last_name, first_name, middle_name = extract_name(full_name) if full_name else (None, None, None)

    return license_no, last_name, first_name, middle_name, birthdate  # Ensure always 5 values

# Streamlit UI
st.title("📝 Philippine Driver's License OCR Extractor")
uploaded_file = st.file_uploader("Upload your Driver's License Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Driver's License", use_container_width=True)
    
    processed_img = preprocess_image(image)

    # Crop a specific area for License No. (adjust as needed)
    img_np = np.array(image)
    height, width, _ = img_np.shape
    license_crop = img_np[int(height*0.6):int(height*0.75), int(width*0.05):int(width*0.5)]
    
    # Run OCR twice: on full image and cropped license number area
    ocr_results = reader.readtext(np.array(processed_img)) + reader.readtext(license_crop)
    
    license_no, last_name, first_name, middle_name, birthdate = extract_info(ocr_results)
    st.subheader("Extracted Information")
    st.write(f"**License No.:** {license_no if license_no else 'Not found'}")
    st.write(f"**Last Name:** {last_name if last_name else 'Not found'}")
    st.write(f"**First Name:** {first_name if first_name else 'Not found'}")
    st.write(f"**Middle Name:** {middle_name if middle_name else 'Not found'}")
    st.write(f"**Date of Birth:** {birthdate if birthdate else 'Not found'}")
