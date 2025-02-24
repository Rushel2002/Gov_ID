import easyocr
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import re

# Initialize EasyOCR reader with English & Filipino
reader = easyocr.Reader(['en', 'tl'], gpu=False)

def preprocess_image(image):
    """Enhance image contrast, denoise, and apply adaptive thresholding."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Apply sharpening filter
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpening_kernel)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
    return binary

def extract_passport_info(ocr_results):
    """Extracts key details from the Philippine Passport."""
    passport_no, surname, given_names, middle_name, birthdate = None, None, None, None, None
    
    # Extract text only, ensuring proper structure
    text_lines = []
    for entry in ocr_results:
        if isinstance(entry, (list, tuple)) and len(entry) > 1:
            text_lines.append(str(entry[1]).upper())
        elif isinstance(entry, str):
            text_lines.append(entry.upper())

    # Function to get text after a specific keyword
    def get_text_after(keywords):
        """Finds the text after a given set of keywords."""
        for keyword in keywords:
            for i, line in enumerate(text_lines):
                if keyword in line and i + 1 < len(text_lines):
                    return text_lines[i + 1]
        return None

    # Extract Passport Number
    passport_pattern = r"P\d{7}[A-Z]?"
    for line in text_lines:
        match = re.search(passport_pattern, line)
        if match:
            passport_no = match.group(0)
            break

    # Extract Surname
    surname = get_text_after(["SURNAME", "APELYIDO"])
    given_names = get_text_after(["GIVEN NAME", "GIVEN NAMES", "PANGALAN"])
    middle_name = get_text_after(["MIDDLE NAME", "PANGGITNANG APELYIDO"])

    # Ensure proper name separation
    if surname:
        surname_parts = surname.split()
        if len(surname_parts) > 1:
            surname = surname_parts[0]  # Assume the first word is the surname
            middle_name = surname_parts[1]  # Assume second word is the middle name if missing

    # Ensure middle name is extracted properly
    if middle_name:
        middle_name = re.sub(r"PANGGITNANG APELYIDO|MIDDLE NAME", "", middle_name, flags=re.IGNORECASE).strip()

    # Handle cases where the middle name is mistakenly included in the given name
    if not middle_name and given_names:
        name_parts = given_names.split()
        if len(name_parts) > 2:
            middle_name = name_parts[-1]  # Assume last word is the middle name
            given_names = " ".join(name_parts[:-1])  # Remove it from given names

    # Extract Date of Birth
    date_pattern = r"\d{2} [A-Z]+ \d{4}"
    for line in text_lines:
        match = re.search(date_pattern, line)
        if match:
            birthdate = match.group(0)
            break

    return passport_no, surname, given_names, middle_name, birthdate

# Streamlit UI
st.title("\U0001F6C2 Philippine Passport OCR Extractor")
uploaded_file = st.file_uploader("Upload your Passport Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    processed_img = preprocess_image(image)
    st.image(image, caption="Uploaded Passport", use_container_width=True)
    
    # Run OCR on both preprocessed and original images
    ocr_results = reader.readtext(processed_img) + reader.readtext(np.array(image))
    
    # Extract Information
    passport_no, surname, given_names, middle_name, birthdate = extract_passport_info(ocr_results)
    
    # Display Results
    st.subheader("Extracted Information")
    st.write(f"**Passport Number:** {passport_no if passport_no else 'Not found'}")
    st.write(f"**Surname:** {surname if surname else 'Not found'}")
    st.write(f"**Given Names:** {given_names if given_names else 'Not found'}")
    st.write(f"**Middle Name:** {middle_name if middle_name else 'Not found'}")
    st.write(f"**Date of Birth:** {birthdate if birthdate else 'Not found'}")
