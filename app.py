import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to load and resize images
def load_and_resize_image(image_path, size=(500, 500)):
    """
    Loads and resizes the image to fit the compositing frame.
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_resized = cv2.resize(img, size)  # Resize image to fit the target size
        return img_resized
    except Exception as e:
        st.error(f"Error loading and resizing image: {e}")
        return None

# Function to add images of people onto the background
def place_people_on_background(background, people_images, positions):
    """
    Places the images of people onto a background at specified positions.
    """
    try:
        for idx, person_image in enumerate(people_images):
            x_offset, y_offset = positions[idx]
            h, w = person_image.shape[:2]
            
            # Ensure that we don't go out of bounds
            if y_offset + h > background.shape[0] or x_offset + w > background.shape[1]:
                continue
            
            # Place person on the background
            background[y_offset:y_offset + h, x_offset:x_offset + w] = person_image

        return background
    except Exception as e:
        st.error(f"Error placing people on background: {e}")
        return None

# Streamlit app for photo compositing
def main():
    st.title("Photo Compositing Workflow")

    # Background upload
    background_file = st.file_uploader("Upload Background Image", type=["jpg", "png"])
    if background_file is not None:
        try:
            background = Image.open(background_file)
            background = np.array(background)
            st.image(background, caption="Background Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error opening background image: {e}")
            return

        # Upload people images
        st.subheader("Upload People Images (Max 10)")
        people_images = []
        for i in range(1, 11):
            person_file = st.file_uploader(f"Upload Person {i}", type=["jpg", "png"])
            if person_file is not None:
                try:
                    person_image = Image.open(person_file)
                    person_image = np.array(person_image)  # Convert PIL image to numpy array
                    people_images.append(person_image)
                except Exception as e:
                    st.error(f"Error opening person image {i}: {e}")
                    return
        
        if people_images:
            # Example positions for people (can be dynamically adjusted)
            positions = [(100, 100), (300, 150), (500, 200), (700, 250),
                         (900, 300), (1100, 350), (1300, 400), (1500, 450),
                         (1700, 500), (1900, 550)]  # Adjust these as needed
            
            # Create final group photo by placing people images on the background
            group_photo = place_people_on_background(background, people_images, positions)

            if group_photo is not None:
                # Display the final group photo
                st.image(group_photo, caption="Generated Group Photo", use_column_width=True)

                # Option to download the generated group photo
                result_path = "generated_group_photo.jpg"
                cv2.imwrite(result_path, cv2.cvtColor(group_photo, cv2.COLOR_RGB2BGR))
                st.download_button(
                    label="Download Group Photo",
                    data=open(result_path, "rb").read(),
                    file_name="group_photo.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()
