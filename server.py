import os

import gdown
import torch
from fastapi import FastAPI
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor

app = FastAPI()

# Define paths and device
model_save_dir = "model"  # Default path to save the model if not defined
os.makedirs(model_save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize feature extractor and tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Define local model checkpoint path and Google Drive file ID for direct download
model_checkpoint_path = os.path.join(model_save_dir, "best_model_300_epochs.pt")
# file_id = "1bwpG3-Jh5d_IM4dYmEsDt39I-bl7n9qn"
file_id = "1N_gmx8tJtlV-UC6uTUIFEEWDwRdWngI-"
url = f"https://drive.google.com/uc?id={file_id}"

# Check if model checkpoint exists locally; if not, attempt to download it
if not os.path.exists(model_checkpoint_path):
    print(
        "Model checkpoint not found locally. Attempting to download from Google Drive..."
    )
    try:
        gdown.download(url, model_checkpoint_path, quiet=False)
        print("Model checkpoint downloaded successfully.")
    except Exception as e:
        print("Failed to download model from Google Drive:", e)

# Load the model and apply the checkpoint
try:
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    model.load_state_dict(
        torch.load(model_checkpoint_path, map_location=device), strict=False
    )
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded and checkpoint applied successfully.")
except Exception as e:
    print("Failed to load model:", e)


def process_image(image_path):
    """
    Load and preprocess an image for the model.
    Args:
        image_path (str): Path to the image file.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
    return image_tensor.to(device)  # Ensure image tensor is on the right device


def generate_caption(image):
    """
    Generate a caption for the input image tensor.
    Args:
        image: Path to the image file.
    Returns:
        str: Generated caption.
    """
    image_tensor = process_image(image)
    with torch.no_grad():
        output = model.generate(pixel_values=image_tensor)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption


@app.post("/predict/")
async def predict(input_data: dict):
    file_path = input_data["file-path"]
    print(file_path)
    result = generate_caption(file_path)
    print(result)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
