import cv2
import numpy as np
import streamlit as st
import easyocr
import re
import tempfile
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to extract details from the PhilHealth ID
def extract_philhealth_details(image_path):
    result = reader.readtext(image_path)
    text_lines = [text[1].upper() for text in result]

    philhealth_number, last_name, first_name, middle_name, birthdate = "Not found", "Not found", "Not found", "Not found", "Not found"

    # Extract PhilHealth Number
    philhealth_pattern = r"\b\d{2}-\d{9}-\d{1}\b"
    for line in text_lines:
        match = re.search(philhealth_pattern, line)
        if match:
            philhealth_number = match.group(0)
            break

    # Extract Name
    name_pattern = re.compile(r"([A-Z]+),\s([A-Z]+)(?:\s([A-Z]+))?")
    for line in text_lines:
        match = name_pattern.search(line)
        if match:
            last_name = match.group(1)
            first_name = match.group(2)
            middle_name = match.group(3) if match.group(3) else "Not found"
            break

    # Extract Date of Birth
    date_pattern = r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s\d{1,2},\s\d{4}"
    for line in text_lines:
        match = re.search(date_pattern, line)
        if match:
            birthdate = match.group(0)
            break
        
    return philhealth_number, last_name, first_name, middle_name, birthdate

# Streamlit UI
st.title("PhilHealth ID OCR Extractor")
st.write("Capture an image using the webcam or upload a PhilHealth ID image.")

# Capture Image from Webcam using Streamlit Camera Input
captured_image = st.camera_input("Take a picture of the PhilHealth ID")

if captured_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_filename = temp_file.name
        image = Image.open(captured_image)

        # ✅ Convert image to RGB if it's RGBA or has an alpha channel
        image = image.convert("RGB")
        image.save(temp_filename, "JPEG")

    st.success("Image captured successfully!")

    # Process the captured image
    philhealth_number, last_name, first_name, middle_name, birthdate = extract_philhealth_details(temp_filename)

    # Display Image & Extracted Information
    st.image(temp_filename, caption="Captured Image", use_container_width=True)
    st.subheader("Extracted Information")
    st.write(f"**PhilHealth Number:** {philhealth_number}")
    st.write(f"**Last Name:** {last_name}")
    st.write(f"**First Name:** {first_name}")
    st.write(f"**Middle Name:** {middle_name}")
    st.write(f"**Date of Birth:** {birthdate}")

# Upload Image Option
uploaded_file = st.file_uploader("Upload a PhilHealth ID image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)

    # ✅ Convert image to RGB if it's RGBA or has an alpha channel
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_filename = temp_file.name
        image.convert("RGB").save(temp_filename, "JPEG")

    # Extract details
    philhealth_number, last_name, first_name, middle_name, birthdate = extract_philhealth_details(temp_filename)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.subheader("Extracted Information")
    st.write(f"**PhilHealth Number:** {philhealth_number}")
    st.write(f"**Last Name:** {last_name}")
    st.write(f"**First Name:** {first_name}")
    st.write(f"**Middle Name:** {middle_name}")
    st.write(f"**Date of Birth:** {birthdate}")