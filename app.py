import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os

# Helper functions
def load_and_resize(image_file, size):
    """
    Load an image from a file and resize it to the given size.
    """
    img = Image.open(image_file)
    img = img.resize(size)
    return np.array(img)

def composite_images(background, foreground_list, output_size=(1024, 1024)):
    """
    Composite multiple foreground images onto the background.
    """
    try:
        # Convert background to OpenCV format
        background = cv2.resize(background, output_size)

        # Ensure background is in RGB format
        if background.shape[-1] == 4:  # If RGBA, convert to RGB
            background = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)

        # Composite all foregrounds
        for fg in foreground_list:
            fg_resized = cv2.resize(fg, output_size)
            if fg_resized.shape[-1] == 4:  # If RGBA, convert to RGB
                fg_resized = cv2.cvtColor(fg_resized, cv2.COLOR_RGBA2RGB)
            background = cv2.addWeighted(background, 0.7, fg_resized, 0.3, 0)

        return background
    except Exception as e:
        st.error(f"Error compositing images: {e}")
        return None

# Streamlit UI
st.title("Photo Compositing Workflow")
st.write("Upload a background image and one or more foreground images to composite them together.")

# File upload inputs
background_file = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
foreground_files = st.file_uploader("Upload Foreground Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if background_file and foreground_files:
    # Load images
    background = load_and_resize(background_file, (1024, 1024))
    foregrounds = [load_and_resize(fg, (1024, 1024)) for fg in foreground_files]

    # Composite images
    result = composite_images(background, foregrounds)

    if result is not None:
        # Display result
        st.image(result, caption="Composited Image", use_column_width=True)

        # Save output
        output_path = "output_composited.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        st.success(f"Composited image saved as {output_path}.")
