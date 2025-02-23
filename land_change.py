import streamlit as st
import numpy as np
from PIL import Image
import cv2

def detect_land_changes(image1_path, image2_path):
    """Detect and display changes between two images."""
    # Load images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Convert images to numpy arrays
    image_array1 = np.array(image1)
    image_array2 = np.array(image2)
    
    # Ensure images are the same size
    if image_array1.shape != image_array2.shape:
        return "Error: Images must be the same size."
    
    # Detect changes
    changes = cv2.absdiff(image_array1, image_array2)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image1, caption="Image 1")
    with col2:
        st.image(image2, caption="Image 2")
    with col3:
        st.image(changes, caption="Changes Detected")
    
    # Calculate change statistics
    change_percent = np.sum(changes > 50) / changes.size * 100
    st.write(f"Changed Area: {change_percent:.2f}%")
    
    return changes

def main():
    st.title("🌍 Land Change Detection")
    st.write("Upload two images to detect changes over time.")

    # File upload
    uploaded_file2 = st.file_uploader("Upload first image", type=['tif', 'png', 'jpg'])
    uploaded_file3 = st.file_uploader("Upload second image", type=['tif', 'png', 'jpg'])

    if uploaded_file2 is not None and uploaded_file3 is not None:
        detect_land_changes(uploaded_file2, uploaded_file3)

if __name__ == "__main__":
    main()