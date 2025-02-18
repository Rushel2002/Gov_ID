import easyocr
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import re

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(image):
    """Enhance image contrast, denoise, and apply adaptive thresholding."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Denoising to remove unnecessary noise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding to improve OCR readability
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

    return binary

def correct_month(text):
    """Fix common OCR mistakes where the first letter of the month is missing."""
    months = [
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
        "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
    ]
    for month in months:
        if text in month:
            return month  # Return the correct month name
    return text  # Return the original if no fix was needed

def extract_info(ocr_results):
    """Extracts key details (ID Number, Name, Date of Birth)."""
    id_number, last_name, first_name, middle_name, suffix, birthdate = None, None, None, None, None, None

    # Extract detected text from OCR results
    text_lines = [text.upper() for _, text, _ in ocr_results]

    # Function to get text after a specific keyword
    def get_text_after(keyword):
        for i, line in enumerate(text_lines):
            if keyword in line:
                return text_lines[i + 1] if i + 1 < len(text_lines) else None
        return None

    # Extract ID number (allowing dashes and spaces)
    id_pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    for line in text_lines:
        id_match = re.search(id_pattern, line)
        if id_match:
            raw_id = re.sub(r"[^0-9]", "", id_match.group(0))  # Remove all non-numeric characters
            if len(raw_id) == 16:  # Ensure correct length before formatting
                id_number = f"{raw_id[:4]}-{raw_id[4:8]}-{raw_id[8:12]}-{raw_id[12:]}"
            break

    # Extract names based on common label variations
    last_name = get_text_after("LAST NAME") or get_text_after("APELYIDO")
    first_name = get_text_after("GIVEN NAMES") or get_text_after("MGA PANGALAN")
    middle_name = get_text_after("MIDDLE NAME") or get_text_after("GITNANG APELYIDO")

    # Extract date of birth using regex for multiple formats
    date_patterns = [
        r"([A-Z]+)\s(\d{1,2}),\s(\d{4})",   # JANUARY 01, 1990
        r"(\d{2})/(\d{2})/(\d{4})",         # 01/01/1990
        r"(\d{4})-(\d{2})-(\d{2})"          # 1990-01-01
    ]

    for line in text_lines:
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                if len(match.groups()) == 3:
                    if pattern == date_patterns[0]:  # "JANUARY 01, 1990"
                        birthdate = f"{correct_month(match.group(1))} {match.group(2)}, {match.group(3)}"
                    elif pattern == date_patterns[1]:  # "01/01/1990"
                        month_name = correct_month([
                            "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
                            "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
                        ][int(match.group(1))-1])  # Convert month number to name
                        birthdate = f"{month_name} {match.group(2)}, {match.group(3)}"
                    elif pattern == date_patterns[2]:  # "1990-01-01"
                        month_name = correct_month([
                            "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
                            "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
                        ][int(match.group(2))-1])  # Convert month number to name
                        birthdate = f"{month_name} {match.group(3)}, {match.group(1)}"
                break
        if birthdate:
            break

    return id_number, last_name, first_name, middle_name, suffix, birthdate

# Streamlit UI
st.title("📝 Philippine National ID OCR Extractor (Enhanced)")
id_type = st.selectbox("Select ID Type", ["National ID", "Passport", "Postal", "Driver's License", "SSS ID", "PhilHealth ID", "TIN ID"])
uploaded_file = st.file_uploader("Upload your National ID Image")

if uploaded_file:
    image = Image.open(uploaded_file)

    # Preprocess image
    processed_img = preprocess_image(image)

    # Run OCR on both original and preprocessed images
    ocr_results = reader.readtext(processed_img) + reader.readtext(np.array(image))

    # Extract text information
    id_number, last_name, first_name, middle_name, suffix, birthdate = extract_info(ocr_results)

    # Draw bounding boxes on the image
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    for (bbox, text, _) in ocr_results:
        box = [tuple(map(int, point)) for point in bbox]
        draw.polygon(box, outline='red')
        draw.text((box[0][0], box[0][1]), text, fill='red')

    # Display the processed image with bounding boxes
    st.image(image_with_boxes, caption="ID with Bounding Boxes", use_container_width=True)

    # Display extracted information
    st.subheader("Extracted Information")
    st.write(f"**ID Type:** {id_type}")  # Displays the selected ID type
    st.write(f"**ID Number:** {id_number if id_number else 'Not found'}")
    st.write(f"**Last Name:** {last_name if last_name else 'Not found'}")
    st.write(f"**First Name:** {first_name if first_name else 'Not found'}")
    st.write(f"**Middle Name:** {middle_name if middle_name else 'Not found'}")
    st.write(f"**Suffix:** {suffix if suffix else 'None'}")
    st.write(f"**Date of Birth:** {birthdate if birthdate else 'Not found'}")
