import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.title("Art Digitization Platform")

# Image upload
uploaded_file = st.file_uploader("Upload an Artwork", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Artwork", use_column_width=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Artwork", use_column_width=True)

    # Metadata extraction
    st.write("Image Metadata:")
    st.write(f"Format: {image.format}")
    st.write(f"Size: {image.size} pixels")
    st.write(f"Mode: {image.mode}")

if uploaded_file:
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Resize image
    resized_image = cv2.resize(image_cv, (800, 800))

    # Denoise image
    denoised_image = cv2.fastNlMeansDenoisingColored(resized_image, None, 10, 10, 7, 21)

    st.write("Enhanced Image:")
    st.image(denoised_image, channels="BGR", use_column_width=True)

