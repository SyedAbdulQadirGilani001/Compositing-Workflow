import cv2
import numpy as np
import os

# Function to load and resize an image
def load_and_resize(image_path, target_size):
    """
    Loads an image and resizes it to the given target size.
    Handles both RGBA and RGB images.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): (width, height) to resize the image.

    Returns:
        np.ndarray: Resized image in RGB format.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with transparency if present

    # If image has 4 channels (RGBA), convert to RGB
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Resize the image
    image = cv2.resize(image, target_size)
    return image

# Function to composite a foreground image onto a background
def composite_images(background_path, foreground_paths, output_path, output_size=(1024, 1024)):
    """
    Composites multiple foreground images onto a background image.

    Args:
        background_path (str): Path to the background image.
        foreground_paths (list): List of paths to foreground images.
        output_path (str): Path to save the composited image.
        output_size (tuple): (width, height) of the final output image.
    """
    try:
        # Load and resize the background image
        background = load_and_resize(background_path, output_size)

        # Loop through all foreground images
        for foreground_path in foreground_paths:
            # Load and resize the foreground image
            foreground = load_and_resize(foreground_path, output_size)

            # Blend the images (adjust alpha values for better blending)
            background = cv2.addWeighted(background, 0.7, foreground, 0.3, 0)

        # Save the composited image
        cv2.imwrite(output_path, background)
        print(f"Composited image saved successfully at {output_path}")

    except Exception as e:
        print(f"Error during compositing: {e}")

# Main function to execute the workflow
def main():
    """
    Main function to execute the photo compositing workflow.
    """
    # Paths to images
    background_path = "background.jpg"  # Replace with your background image path
    foreground_paths = ["person1.png", "person2.png", "person3.png"]  # List of foreground images
    output_path = "composited_image.jpg"  # Output image path

    # Check if all files exist
    if not os.path.exists(background_path):
        print(f"Error: Background image not found at {background_path}")
        return

    for path in foreground_paths:
        if not os.path.exists(path):
            print(f"Error: Foreground image not found at {path}")
            return

    # Run the compositing function
    composite_images(background_path, foreground_paths, output_path)

# Run the script
if __name__ == "__main__":
    main()
