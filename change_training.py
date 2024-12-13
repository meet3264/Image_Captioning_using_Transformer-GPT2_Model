import json
import os
import random

import torch
from nltk.translate.bleu_score import sentence_bleu  # For BLEU reward calculation
from PIL import Image
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


def load_annotations(inputs, outputs):
    annotation_file = inputs["annotation_file"]
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    captions_dict = {
        annotation["image_id"]: annotation["caption"]
        for annotation in annotations["annotations"]
    }
    outputs["captions_dict"] = captions_dict


def process_image(inputs, outputs):
    image = Image.open(inputs["image_path"]).convert("RGB")
    image_tensor = inputs["feature_extractor"](
        images=image, return_tensors="pt"
    ).pixel_values
    outputs["image_tensor"] = image_tensor.to(inputs["device"])


def process_caption(inputs, outputs):
    caption_tensor = inputs["tokenizer"](
        inputs["caption"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=inputs["max_length"],
    ).input_ids
    outputs["caption_tensor"] = caption_tensor.to(inputs["device"])


def configure_model(inputs, outputs):
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.to(inputs["device"])
    optimizer = AdamW(model.parameters(), lr=inputs["learning_rate"])
    outputs["model"] = model
    outputs["feature_extractor"] = feature_extractor
    outputs["tokenizer"] = tokenizer
    outputs["optimizer"] = optimizer


def calculate_bleu(reference, candidate):
    # Calculate BLEU score as a reward metric
    return sentence_bleu([reference.split()], candidate.split())


def reinforce_loss(predicted_logits, target_ids, rewards):
    # Calculate Reinforce loss with policy gradients
    log_probs = F.log_softmax(predicted_logits, dim=-1)
    seq_len = target_ids.shape[1]
    loss = 0
    for i in range(seq_len):
        token_log_prob = log_probs[:, i, target_ids[:, i]]
        loss -= rewards[i] * token_log_prob
    return loss.mean()


def train_model(inputs, outputs):
    model = inputs["model"]
    feature_extractor = inputs["feature_extractor"]
    tokenizer = inputs["tokenizer"]
    optimizer = inputs["optimizer"]
    image_dir = inputs["image_dir"]
    captions_dict = inputs["captions_dict"]
    device = inputs["device"]
    model_save_dir = inputs["model_save_dir"]
    num_epochs = inputs["num_epochs"]
    batch_size = inputs["batch_size"]
    num_samples_per_epoch = inputs["num_samples_per_epoch"]
    saved_model_paths = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        sample_keys = random.sample(list(captions_dict.keys()), num_samples_per_epoch)

        for batch_idx in range(0, num_samples_per_epoch, batch_size):
            batch_keys = sample_keys[batch_idx : batch_idx + batch_size]
            images, captions, rewards = [], [], []

            for image_id in batch_keys:
                image_path = os.path.join(
                    image_dir, f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
                )
                caption = captions_dict[image_id]

                # Process image
                image_output = {}
                process_image(
                    {
                        "image_path": image_path,
                        "feature_extractor": feature_extractor,
                        "device": device,
                    },
                    image_output,
                )
                images.append(image_output["image_tensor"])

                # Process caption
                caption_output = {}
                process_caption(
                    {
                        "caption": caption,
                        "tokenizer": tokenizer,
                        "device": device,
                        "max_length": 50,
                    },
                    caption_output,
                )
                captions.append(caption_output["caption_tensor"])

            # Concatenate images and captions
            images = torch.cat(images)
            captions = torch.cat(captions)

            # Forward pass
            outputs = model(pixel_values=images, labels=captions)
            predicted_ids = outputs.logits.argmax(-1)

            # Generate text and calculate reward
            for i in range(len(batch_keys)):
                reference_caption = captions_dict[batch_keys[i]]
                generated_caption = tokenizer.decode(
                    predicted_ids[i], skip_special_tokens=True
                )
                reward = calculate_bleu(reference_caption, generated_caption)
                rewards.append(reward)

            # Reinforce loss
            rewards = torch.tensor(rewards).to(device)
            loss = reinforce_loss(outputs.logits, captions, rewards)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch {epoch + 1}, Batch {batch_idx // batch_size + 1}/{num_samples_per_epoch // batch_size} - Loss: {loss.item()}"
            )

        print(
            f"Epoch {epoch + 1} - Average Loss: {total_loss / (num_samples_per_epoch / batch_size)}"
        )

        # Save model checkpoint
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(
            model_save_dir, f"vit_gpt2_image_captioning_SCST_epoch_{epoch + 1}.pt"
        )
        torch.save(model.state_dict(), save_path)
        saved_model_paths.append(save_path)
        print(f"Model saved after epoch {epoch + 1} at {save_path}")

    outputs["saved_model_paths"] = saved_model_paths


if __name__ == "__main__":
    inputs = {
        "annotation_file": "coco_dataset/annotations/captions_train2014.json",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "learning_rate": 5e-5,
        "image_dir": "coco_dataset/images/train2014",
        "model_save_dir": "model",
        "num_epochs": 10,
        "batch_size": 50,
        "num_samples_per_epoch": 5000,
    }
    outputs = {}
    load_annotations({"annotation_file": inputs["annotation_file"]}, outputs)
    inputs["captions_dict"] = outputs["captions_dict"]
    configure_model(
        {"device": inputs["device"], "learning_rate": inputs["learning_rate"]}, outputs
    )
    inputs["model"] = outputs["model"]
    inputs["feature_extractor"] = outputs["feature_extractor"]
    inputs["tokenizer"] = outputs["tokenizer"]
    inputs["optimizer"] = outputs["optimizer"]
    train_model(inputs, outputs)
