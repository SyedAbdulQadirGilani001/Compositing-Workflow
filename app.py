import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Helper function to resize and maintain aspect ratio
def resize_with_aspect_ratio(image, width=None, height=None):
    """
    Resize an image while maintaining its aspect ratio.
    """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Helper function for compositing images
def composite_images(background, foreground_list, positions):
    """
    Place foreground images at specified positions on the background.
    """
    for fg, (x, y, scale) in zip(foreground_list, positions):
        # Resize foreground based on scale
        fg_resized = resize_with_aspect_ratio(fg, width=int(fg.shape[1] * scale))
        
        # Get dimensions
        fg_h, fg_w = fg_resized.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Check if position fits in the background
        if y + fg_h > bg_h or x + fg_w > bg_w:
            st.warning(f"Foreground at ({x},{y}) exceeds background bounds. Skipping.")
            continue
        
        # Create mask for blending
        if fg_resized.shape[2] == 4:  # RGBA
            alpha = fg_resized[:, :, 3] / 255.0
            fg_resized = fg_resized[:, :, :3]  # Drop alpha channel
        else:
            alpha = np.ones((fg_h, fg_w), dtype=float)
        
        # Composite the images
        for c in range(3):  # For each color channel
            background[y:y+fg_h, x:x+fg_w, c] = (
                alpha * fg_resized[:, :, c] +
                (1 - alpha) * background[y:y+fg_h, x:x+fg_w, c]
            )
    return background

# Streamlit app
st.title("Enhanced Photo Compositing Workflow")
st.write("Upload a background and people images to create a composite. Ensure proper placement and scaling.")

# File uploads
background_file = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
foreground_files = st.file_uploader("Upload Foreground Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if background_file and foreground_files:
    # Load background
    background = np.array(Image.open(background_file).convert("RGB"))
    background = resize_with_aspect_ratio(background, width=1024)

    # Load foreground images
    foregrounds = [np.array(Image.open(f).convert("RGBA")) for f in foreground_files]

    # Specify positions (can be dynamic based on user input)
    st.write("Specify positions (x, y) and scaling factor for each person:")
    positions = []
    for i, fg in enumerate(foregrounds):
        x = st.number_input(f"Person {i + 1}: X Position", min_value=0, max_value=background.shape[1], value=50 * (i + 1))
        y = st.number_input(f"Person {i + 1}: Y Position", min_value=0, max_value=background.shape[0], value=50)
        scale = st.slider(f"Person {i + 1}: Scale", min_value=0.1, max_value=1.0, value=0.5)
        positions.append((x, y, scale))
    
    # Composite images
    result = composite_images(background, foregrounds, positions)
    
    # Display result
    st.image(result, caption="Composited Image", use_column_width=True)

    # Save result
    result_path = "composited_image.jpg"
    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    st.success(f"Composited image saved as {result_path}.")
