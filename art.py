import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
from diffusers import StableDiffusionPipeline
import torch

# ======= Step 1: Art Digitization =======
def upload_and_enhance_image():
    st.title("Art Digitization")
    uploaded_file = st.file_uploader("Upload an Artwork", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Artwork", use_column_width=True)
        
        # Display metadata
        st.write(f"Image Format: {image.format}")
        st.write(f"Image Size: {image.size}")
        st.write(f"Image Mode: {image.mode}")
        
        # Enhance image
        resized_image = image.resize((512, 512))
        enhanced_image = ImageOps.autocontrast(resized_image)
        st.image(enhanced_image, caption="Enhanced Artwork", use_column_width=True)

# ======= Step 2: Style Transfer =======
@st.cache_resource
def load_style_transfer_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return tf.convert_to_tensor(image, dtype=tf.float32)[tf.newaxis, ...]

def perform_style_transfer():
    st.title("Style Transfer")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])
    
    if content_file and style_file:
        content_image = Image.open(content_file)
        style_image = Image.open(style_file)
        st.image(content_image, caption="Content Image", use_column_width=True)
        st.image(style_image, caption="Style Image", use_column_width=True)
        
        # Style transfer
        model = load_style_transfer_model()
        content_tensor = preprocess_image(content_image)
        style_tensor = preprocess_image(style_image)
        stylized_image = model(content_tensor, style_tensor)[0]
        stylized_image = tf.image.convert_image_dtype(stylized_image[0], dtype=tf.uint8)
        
        st.image(stylized_image.numpy(), caption="Stylized Image", use_column_width=True)

# ======= Step 3: Custom Art Generation =======
@st.cache_resource
def load_generation_pipeline():
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipeline.to("cuda")
    return pipeline

def generate_custom_art():
    st.title("Custom Art Generator")
    prompt = st.text_input("Enter a theme or description for your art:")
    if st.button("Generate Art"):
        if prompt:
            with st.spinner("Generating artwork..."):
                pipeline = load_generation_pipeline()
                result = pipeline(prompt).images[0]
                st.image(result, caption="Generated Art", use_column_width=True)
        else:
            st.warning("Please enter a prompt.")

# ======= Main Application =======
def main():
    st.sidebar.title("Art Creativity Platform")
    options = ["Art Digitization", "Style Transfer", "Custom Art Generation"]
    choice = st.sidebar.selectbox("Select a Feature", options)

    if choice == "Art Digitization":
        upload_and_enhance_image()
    elif choice == "Style Transfer":
        perform_style_transfer()
    elif choice == "Custom Art Generation":
        generate_custom_art()

if __name__ == "__main__":
    main()
