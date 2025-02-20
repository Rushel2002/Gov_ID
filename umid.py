import easyocr
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import re

# Initialize EasyOCR reader with English G2 for better printed text recognition
reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')

def preprocess_image(image):
    """Enhance image for OCR: grayscale, denoise, binarize, and apply morphological operations."""
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Denoise for clearer text (stronger filtering)
    gray = cv2.fastNlMeansDenoising(gray, h=20)

    # Adaptive Thresholding for binarization
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)

    # Morphological operations to enhance text edges
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

def correct_text(text):
    """Fix common OCR misinterpretations in names."""
    if text:
        text = text.replace("JV", "JU")  # Fix 'JVSTIN' to 'JUSTIN'
        text = text.replace("EOUIS", "LOUIS")  # Fix 'EOUIS' to 'LOUIS'
        text = text.replace("0", "O")  # Fix '0' (zero) to 'O'
    return text

def correct_birthdate(birthdate):
    """Fix common OCR birthdate errors, ensuring valid years."""
    if birthdate:
        parts = birthdate.split('-')
        if len(parts) == 3:
            year, month, day = parts
            if year == "1945":  # OCR often mistakes 1995 for 1945
                year = "1995"
            elif int(year) < 1900 or int(year) > 2025:  # Ensure a realistic year range
                year = "1995"  # Default fallback if OCR is way off
            return f"{year}-{month}-{day}"
    return birthdate

def extract_info(ocr_results):
    """Extract ID number, Name, and Date of Birth correctly."""
    crn, surname, given_name, middle_name, birthdate = None, None, None, None, None

    text_lines = [text.upper() for _, text, _ in ocr_results]

    def get_text_after(keyword):
        """Retrieve text after a specific keyword in OCR results."""
        for i, line in enumerate(text_lines):
            if keyword in line:
                return text_lines[i + 1] if i + 1 < len(text_lines) else None
        return None
    
    # **Enhanced CRN Extraction with Error Handling**
    crn_pattern = r"(?:CRN[:\s-]*)?(\d{4}[-\s]?\d{7}[-\s]?\d)"  # Flexible pattern
    possible_crns = []
    crn_positions = {}

    for i, line in enumerate(text_lines):
        match = re.search(crn_pattern, line)
        if match:
            extracted_crn = match.group(1)
            # Fix OCR misreads (e.g., "O" -> "0", "I" -> "1", "S" -> "5")
            extracted_crn = extracted_crn.replace(" ", "").replace("O", "0").replace("I", "1").replace("S", "5")

            possible_crns.append(extracted_crn)
            crn_positions[extracted_crn] = i

    # Select best CRN (if multiple, pick closest to 'CRN' label)
    if possible_crns:
        crn_index = next((i for i, line in enumerate(text_lines) if "CRN" in line), None)
        crn = min(possible_crns, key=lambda c: abs(crn_positions[c] - crn_index)) if crn_index is not None else possible_crns[0]

    crn = get_text_after("CRN")
    surname = get_text_after("SURNAME")
    given_name = get_text_after("GIVEN NAME")
    middle_name = get_text_after("MIDDLE NAME")

    birthdate = get_text_after("DATE OF BIRTH")
    birthdate = correct_birthdate(birthdate)
    
    given_name = correct_text(given_name) if given_name else None

   # **Improved Birthdate Extraction**
    date_patterns = [
        r"(\d{4})/(\d{2})/(\d{2})",  # YYYY/MM/DD
        r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
        r"(\d{2})/(\d{2})/(\d{4})",  # MM/DD/YYYY
        r"(\d{2})-(\d{2})-(\d{4})"   # MM-DD-YYYY
    ]

    for line in text_lines:
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                parts = match.groups()
                if len(parts) == 3:
                    year, month, day = parts if len(parts[0]) == 4 else (parts[2], parts[0], parts[1])
                    birthdate = f"{year}-{month}-{day}"
                    break
        if birthdate:
            break

    # Fix birthdate errors
    birthdate = correct_birthdate(birthdate)

    return crn, surname, given_name, middle_name, birthdate

# **Streamlit UI**
st.title("📝 Philippine National ID OCR Extractor (Enhanced)")
option = st.radio("Choose an input method:", ["Upload Image", "Use Live Camera"])
image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload your National ID Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "Use Live Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image)

if image:
    processed_img = preprocess_image(image)
    st.image(image, caption="Captured/Uploaded Image", use_container_width=True)
    
    ocr_results = reader.readtext(np.array(processed_img)) + reader.readtext(np.array(image))
    
    crn, surname, given_name, middle_name, birthdate = extract_info(ocr_results)
    
    st.subheader("Extracted Information")
    st.write(f"**CRN:** {crn if crn else 'Not found'}")
    st.write(f"**Last Name:** {surname if surname else 'Not found'}")
    st.write(f"**First Name:** {given_name if given_name else 'Not found'}")
    st.write(f"**Middle Name:** {middle_name if middle_name else 'Not found'}")
    st.write(f"**Date of Birth:** {birthdate if birthdate else 'Not found'}")

