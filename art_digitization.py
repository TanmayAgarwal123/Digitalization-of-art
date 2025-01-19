import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from diffusers import StableDiffusionPipeline
import torch
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

def apply_style(content_image, style_image):
    model = load_model()

    # Preprocess the images
    content_image = tf.image.convert_image_dtype(content_image, tf.float32)
    content_image = tf.image.resize(content_image, (256, 256))
    content_image = tf.expand_dims(content_image, axis=0)

    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    style_image = tf.image.resize(style_image, (256, 256))
    style_image = tf.expand_dims(style_image, axis=0)

    # Apply style transfer
    stylized_image = model(content_image, style_image)[0]

    return tf.squeeze(stylized_image).numpy()


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

# Upload style image
style_file = st.file_uploader("Upload a Style Image", type=["jpg", "jpeg", "png"])

if uploaded_file and style_file:
    content_image = np.array(Image.open(uploaded_file))
    style_image = np.array(Image.open(style_file))

    st.write("Content Image:")
    st.image(content_image, use_column_width=True)

    st.write("Style Image:")
    st.image(style_image, use_column_width=True)

    # Apply style transfer
    st.write("Stylized Image:")
    stylized_image = apply_style(content_image, style_image)
    st.image(stylized_image, use_column_width=True)
