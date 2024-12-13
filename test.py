import os

import gdown
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


def load_model(model_save_dir, device):
    """
    Load the trained model, tokenizer, and feature extractor.
    Args:
        model_save_dir (str): Directory where the model is saved.
        device (torch.device): Device to load the model on.
    Returns:
        model, feature_extractor, tokenizer
    """
    # Initialize feature extractor and tokenizer
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Define local model checkpoint path and Google Drive URL
    model_checkpoint_path = os.path.join(model_save_dir, "best_model_300_epochs.pt")
    file_id = "1bwpG3-Jh5d_IM4dYmEsDt39I-bl7n9qn"
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
            return None, None, None

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
        return None, None, None

    return model, feature_extractor, tokenizer


def process_image(image_path, feature_extractor, device):
    """
    Load and preprocess an image for the model.
    Args:
        image_path (str): Path to the image file.
        feature_extractor (ViTFeatureExtractor): Feature extractor for the model.
        device (torch.device): Device to process the image on.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
        return image_tensor.to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def generate_caption(model, image_tensor, tokenizer):
    """
    Generate a caption for the input image tensor.
    Args:
        model: The trained VisionEncoderDecoderModel.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        tokenizer (AutoTokenizer): Tokenizer for the model.
    Returns:
        str: Generated caption.
    """
    if image_tensor is None:
        return "Error: Image processing failed."
    with torch.no_grad():
        output = model.generate(pixel_values=image_tensor)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption


if __name__ == "__main__":
    # Define paths and device
    model_save_dir = "model"  # Path to your saved model
    image_path = "uploaded_images/COCO_val2014_000000000294.jpg"  # Replace with your test image path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model, feature extractor, and tokenizer
    model, feature_extractor, tokenizer = load_model(model_save_dir, device)

    # Ensure model is loaded successfully before proceeding
    if model is not None and feature_extractor is not None and tokenizer is not None:
        # Process the image
        image_tensor = process_image(image_path, feature_extractor, device)

        # Generate caption
        caption = generate_caption(model, image_tensor, tokenizer)

        # Print the generated caption
        print("Generated Caption:", caption)
    else:
        print("Failed to load model or dependencies.")
