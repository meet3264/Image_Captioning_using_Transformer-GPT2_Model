import json
import os
from collections import Counter
from test import generate_caption, load_model, process_image

import nltk
import pandas as pd
import torch
from nltk import ngrams
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image
from rouge import Rouge

nltk.download("punkt")


def calculate_bleu_scores(generated_captions, reference_captions):
    """
    Calculate both original-style BLEU and individual BLEU-1 to BLEU-4 scores
    """
    smooth_fn = SmoothingFunction().method1

    # For original style (single) BLEU score
    total_bleu = 0

    # For individual BLEU-n scores - store best score for each caption
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []

    for generated in generated_captions:
        best_bleu = 0
        best_bleu_1 = 0
        best_bleu_2 = 0
        best_bleu_3 = 0
        best_bleu_4 = 0

        generated_tokens = nltk.word_tokenize(generated.lower())

        # Calculate best scores across all reference sets
        for references in reference_captions:
            reference_tokens_list = [
                nltk.word_tokenize(ref.lower()) for ref in references
            ]

            # Original style BLEU (equal weights)
            bleu_score = sentence_bleu(
                reference_tokens_list,
                generated_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth_fn,
            )
            best_bleu = max(best_bleu, bleu_score)

            # Calculate individual BLEU-n scores with cumulative weights
            bleu_1 = sentence_bleu(
                reference_tokens_list,
                generated_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=smooth_fn,
            )
            bleu_2 = sentence_bleu(
                reference_tokens_list,
                generated_tokens,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smooth_fn,
            )
            bleu_3 = sentence_bleu(
                reference_tokens_list,
                generated_tokens,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=smooth_fn,
            )
            bleu_4 = sentence_bleu(
                reference_tokens_list,
                generated_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth_fn,
            )

            # Update best scores for this caption
            best_bleu_1 = max(best_bleu_1, bleu_1)
            best_bleu_2 = max(best_bleu_2, bleu_2)
            best_bleu_3 = max(best_bleu_3, bleu_3)
            best_bleu_4 = max(best_bleu_4, bleu_4)

        # Add best scores for this caption to our lists
        total_bleu += best_bleu
        bleu_1_scores.append(best_bleu_1)
        bleu_2_scores.append(best_bleu_2)
        bleu_3_scores.append(best_bleu_3)
        bleu_4_scores.append(best_bleu_4)

    # Calculate averages
    count = len(generated_captions)
    original_bleu = total_bleu / count if count > 0 else 0

    bleu_n_scores = {
        "BLEU-1": sum(bleu_1_scores) / count if count > 0 else 0,
        "BLEU-2": sum(bleu_2_scores) / count if count > 0 else 0,
        "BLEU-3": sum(bleu_3_scores) / count if count > 0 else 0,
        "BLEU-4": sum(bleu_4_scores) / count if count > 0 else 0,
    }

    return original_bleu, bleu_n_scores


def calculate_rouge(generated_captions, reference_captions):
    rouge = Rouge()
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for generated in generated_captions:
        best_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

        for references in reference_captions:
            for reference in references:
                scores = rouge.get_scores(generated, reference)[0]
                best_scores["rouge-1"] = max(
                    best_scores["rouge-1"], scores["rouge-1"]["f"]
                )
                best_scores["rouge-2"] = max(
                    best_scores["rouge-2"], scores["rouge-2"]["f"]
                )
                best_scores["rouge-l"] = max(
                    best_scores["rouge-l"], scores["rouge-l"]["f"]
                )

        rouge_scores["rouge1"].append(best_scores["rouge-1"])
        rouge_scores["rouge2"].append(best_scores["rouge-2"])
        rouge_scores["rougeL"].append(best_scores["rouge-l"])

    average_scores = {
        key: sum(value) / len(value) for key, value in rouge_scores.items()
    }

    return average_scores


def calculate_meteor(generated_captions, reference_captions):
    total_meteor = 0
    count = len(generated_captions)

    for generated in generated_captions:
        best_meteor = 0
        generated_tokens = nltk.word_tokenize(generated.lower())

        for references in reference_captions:
            reference_tokens_list = [
                nltk.word_tokenize(ref.lower()) for ref in references
            ]
            meteor = meteor_score(reference_tokens_list, generated_tokens)
            best_meteor = max(best_meteor, meteor)

        total_meteor += best_meteor

    return total_meteor / count if count > 0 else 0


if __name__ == "__main__":
    model_save_dir = "model"
    validation_data_file = "flickr30k_images/results.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_data = pd.read_csv(validation_data_file, delimiter="|")

    model, feature_extractor, tokenizer = load_model(model_save_dir, device)

    generated_captions = []
    reference_captions = {}

    unique_images = set()

    # Process validation data
    for i, row in validation_data.iterrows():
        image_id = row[0]
        reference_caption = row[2]

        if image_id not in reference_captions:
            reference_captions[image_id] = []

        reference_captions[image_id].append(reference_caption)

        if image_id in unique_images:
            continue

        image_path = f"flickr30k_images/flickr30k_images/{image_id}"
        image_tensor = process_image(image_path, feature_extractor, device)
        caption = generate_caption(model, image_tensor, tokenizer)
        generated_captions.append(caption)

        print(f"Image ID: {image_id}, Generated Caption: {caption}")
        unique_images.add(image_id)

        if len(unique_images) >= 2000:  # Set to your desired number of images
            break

    # Calculate metrics
    original_bleu, bleu_n_scores = calculate_bleu_scores(
        generated_captions, list(reference_captions.values())
    )
    meteor_score_avg = calculate_meteor(
        generated_captions, list(reference_captions.values())
    )
    rouge_scores = calculate_rouge(
        generated_captions, list(reference_captions.values())
    )

    print("\nOriginal-style BLEU Score:", original_bleu)
    print("\nIndividual BLEU Scores:")
    for key, value in bleu_n_scores.items():
        print(f"{key}: {value}")
    print("\nMETEOR Score:", meteor_score_avg)
    print("\nROUGE Scores:", rouge_scores)


# import json
# import os
# from collections import Counter
# from functools import lru_cache

# import nltk
# import pandas as pd
# import torch
# import torch.cuda.amp  # for automatic mixed precision
# from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# from nltk.translate.meteor_score import meteor_score
# from PIL import Image
# from rouge import Rouge
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# nltk.download("punkt", quiet=True)


# class ImageCaptionDataset(Dataset):
#     def __init__(self, image_ids, image_dir, feature_extractor):
#         self.image_ids = image_ids
#         self.image_dir = image_dir
#         self.feature_extractor = feature_extractor

#     def __len__(self):
#         return len(self.image_ids)

#     def __getitem__(self, idx):
#         image_id = self.image_ids[idx]
#         image_path = os.path.join(self.image_dir, image_id)
#         image = Image.open(image_path).convert("RGB")
#         return (
#             self.feature_extractor(images=image, return_tensors="pt")["pixel_values"][
#                 0
#             ],
#             image_id,
#         )


# def collate_fn(batch):
#     images = torch.stack([item[0] for item in batch])
#     image_ids = [item[1] for item in batch]
#     return images, image_ids


# @torch.no_grad()
# def generate_captions_batch(model, image_batch, tokenizer, device, max_length=32):
#     """Generate captions for a batch of images using GPU acceleration"""
#     outputs = model.generate(
#         image_batch,
#         max_length=max_length,
#         num_beams=4,
#         length_penalty=1.0,
#         early_stopping=True,
#         pad_token_id=tokenizer.pad_token_id,
#         bos_token_id=tokenizer.bos_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return captions


# @lru_cache(maxsize=1024)
# def tokenize_and_cache(text):
#     """Cache tokenization results"""
#     return nltk.word_tokenize(text.lower())


# def calculate_metrics_gpu(generated_captions, reference_captions):
#     """Calculate all metrics efficiently using GPU where possible"""
#     # Pre-tokenize all captions
#     tokenized_generated = [tokenize_and_cache(cap) for cap in generated_captions]
#     tokenized_refs = {
#         img_id: [tokenize_and_cache(ref) for ref in refs]
#         for img_id, refs in reference_captions.items()
#     }

#     # Calculate BLEU scores
#     smooth_fn = SmoothingFunction().method1
#     bleu_scores = {"BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0}
#     total_bleu = 0

#     # Calculate ROUGE scores
#     rouge = Rouge()
#     rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

#     # Calculate METEOR scores
#     total_meteor = 0

#     for i, gen_tokens in enumerate(
#         tqdm(tokenized_generated, desc="Calculating metrics")
#     ):
#         # BLEU calculation
#         best_bleu_scores = [0] * 5  # For original and BLEU-1 to BLEU-4

#         # ROUGE preparation
#         generated = generated_captions[i]
#         best_rouge_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

#         # Process all references for this caption
#         for refs in tokenized_refs.values():
#             # BLEU scores
#             bleu_original = sentence_bleu(
#                 refs,
#                 gen_tokens,
#                 weights=(0.25, 0.25, 0.25, 0.25),
#                 smoothing_function=smooth_fn,
#             )
#             bleu_1 = sentence_bleu(
#                 refs, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn
#             )
#             bleu_2 = sentence_bleu(
#                 refs, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn
#             )
#             bleu_3 = sentence_bleu(
#                 refs,
#                 gen_tokens,
#                 weights=(0.33, 0.33, 0.33, 0),
#                 smoothing_function=smooth_fn,
#             )
#             bleu_4 = sentence_bleu(
#                 refs,
#                 gen_tokens,
#                 weights=(0.25, 0.25, 0.25, 0.25),
#                 smoothing_function=smooth_fn,
#             )

#             best_bleu_scores = [
#                 max(best_bleu_scores[i], score)
#                 for i, score in enumerate(
#                     [bleu_original, bleu_1, bleu_2, bleu_3, bleu_4]
#                 )
#             ]

#             # METEOR score
#             meteor = meteor_score(refs, gen_tokens)
#             total_meteor = max(total_meteor, meteor)

#             # ROUGE scores
#             for ref_tokens in refs:
#                 try:
#                     rouge_score = rouge.get_scores(generated, " ".join(ref_tokens))[0]
#                     best_rouge_scores["rouge-1"] = max(
#                         best_rouge_scores["rouge-1"], rouge_score["rouge-1"]["f"]
#                     )
#                     best_rouge_scores["rouge-2"] = max(
#                         best_rouge_scores["rouge-2"], rouge_score["rouge-2"]["f"]
#                     )
#                     best_rouge_scores["rouge-l"] = max(
#                         best_rouge_scores["rouge-l"], rouge_score["rouge-l"]["f"]
#                     )
#                 except Exception:
#                     continue

#         # Accumulate scores
#         total_bleu += best_bleu_scores[0]
#         bleu_scores["BLEU-1"] += best_bleu_scores[1]
#         bleu_scores["BLEU-2"] += best_bleu_scores[2]
#         bleu_scores["BLEU-3"] += best_bleu_scores[3]
#         bleu_scores["BLEU-4"] += best_bleu_scores[4]

#         rouge_scores["rouge1"].append(best_rouge_scores["rouge-1"])
#         rouge_scores["rouge2"].append(best_rouge_scores["rouge-2"])
#         rouge_scores["rougeL"].append(best_rouge_scores["rouge-l"])

#     # Calculate final averages
#     count = len(generated_captions)
#     original_bleu = total_bleu / count
#     for key in bleu_scores:
#         bleu_scores[key] /= count

#     meteor_score_avg = total_meteor / count

#     rouge_final = {
#         key: sum(scores) / len(scores) for key, scores in rouge_scores.items()
#     }

#     return original_bleu, bleu_scores, meteor_score_avg, rouge_final


# if __name__ == "__main__":
#     # Configuration
#     model_save_dir = "model"
#     validation_data_file = "flickr30k_images/results.csv"
#     image_dir = "flickr30k_images/flickr30k_images"
#     batch_size = 32  # Adjust based on your GPU memory
#     max_images = 1000

#     # Set up GPU device and enable automatic mixed precision
#     device = torch.device("cuda")
#     scaler = torch.cuda.amp.GradScaler()

#     # Load data and model
#     print("Loading model and data...")
#     validation_data = pd.read_csv(validation_data_file, delimiter="|")
#     model, feature_extractor, tokenizer = load_model(model_save_dir, device)
#     model.eval()  # Set model to evaluation mode

#     # Prepare reference captions
#     reference_captions = {}
#     for i, row in validation_data.iterrows():
#         image_id = row[0]
#         if image_id not in reference_captions:
#             reference_captions[image_id] = []
#         reference_captions[image_id].append(row[2])

#     # Create dataset and dataloader
#     image_ids = list(reference_captions.keys())[:max_images]
#     dataset = ImageCaptionDataset(image_ids, image_dir, feature_extractor)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         collate_fn=collate_fn,
#         pin_memory=True,
#     )

#     # Generate captions
#     generated_captions = []
#     print("\nGenerating captions...")
#     for images, batch_image_ids in tqdm(dataloader, desc="Processing batches"):
#         images = images.to(device)
#         with torch.cuda.amp.autocast():  # Enable automatic mixed precision
#             batch_captions = generate_captions_batch(model, images, tokenizer, device)
#             generated_captions.extend(batch_captions)

#     # Calculate metrics
#     print("\nCalculating evaluation metrics...")
#     original_bleu, bleu_scores, meteor_score_avg, rouge_scores = calculate_metrics_gpu(
#         generated_captions, reference_captions
#     )

#     # Print results
#     print("\nResults:")
#     print("Original-style BLEU Score:", original_bleu)
#     print("\nIndividual BLEU Scores:")
#     for key, value in bleu_scores.items():
#         print(f"{key}: {value}")
#     print("\nMETEOR Score:", meteor_score_avg)
#     print("\nROUGE Scores:", rouge_scores)
