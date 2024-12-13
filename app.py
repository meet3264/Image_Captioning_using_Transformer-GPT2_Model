import os

import requests
import streamlit as st
from PIL import Image


# Function to save the uploaded image to a specific directory
def save_uploaded_image(uploaded_file):
    # Specify the directory where you want to save the images
    save_dir = "uploaded_images"

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the file name and save it
    file_path = os.path.join(save_dir, uploaded_file.name)

    # Write the image to the specified path
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Ensure the buffer is written correctly

    return file_path


# Streamlit application
st.title("Image Caption Generator")
st.write("Upload an image to generate a caption.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image to a specific directory
    file_path = save_uploaded_image(uploaded_image)

    # Generate caption button
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            api_url = "http://127.0.0.1:8000/predict/"
            try:
                response = requests.post(api_url, json={"file-path": file_path})
                response.raise_for_status()  # Raise an error for bad responses
                caption = response.json().get("result", "Caption not found")
                st.success(f"Generated Caption: {caption}")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred: {http_err}")  # Handle HTTP errors
            except requests.exceptions.ConnectionError:
                st.error(
                    "Error: Unable to connect to the server. Please ensure it is running."
                )
            except requests.exceptions.Timeout:
                st.error("Error: The request timed out. Please try again later.")
            except requests.exceptions.RequestException as err:
                st.error(f"An error occurred: {err}")  # Handle other exceptions

            # Optionally delete the file after processing
            os.remove(file_path)

# Optional: Add footer or other UI elements
st.write(
    "This application uses a Vision-Transformer and GPT-2 model for image captioning."
)
