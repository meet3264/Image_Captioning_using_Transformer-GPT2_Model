import json
import os
from collections import Counter
from test import generate_caption, load_model, process_image

import nltk
import torch
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
from rouge import Rouge

nltk.download("punkt")


def calculate_ngram_metrics(generated_captions, reference_captions, n=2):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0

    for generated in generated_captions:
        best_precision = 0
        best_recall = 0
        best_f1 = 0

        for references in reference_captions:
            # Tokenize captions
            generated_tokens = nltk.word_tokenize(generated)

            # Get n-grams for generated
            generated_ngrams = Counter(ngrams(generated_tokens, n))
            overlap = 0

            for reference in references:
                reference_tokens = nltk.word_tokenize(reference)
                reference_ngrams = Counter(ngrams(reference_tokens, n))
                overlap += sum((generated_ngrams & reference_ngrams).values())

                # Calculate precision and recall
                precision = (
                    overlap / len(generated_ngrams) if len(generated_ngrams) > 0 else 0
                )
                recall = (
                    overlap / len(reference_ngrams) if len(reference_ngrams) > 0 else 0
                )

                # Calculate F1 Score
                f1 = (
                    (2 * precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                best_precision = max(best_precision, precision)
                best_recall = max(best_recall, recall)
                best_f1 = max(best_f1, f1)

        total_precision += best_precision
        total_recall += best_recall
        total_f1 += best_f1
        count += 1

    return {
        "Precision": total_precision / count,
        "Recall": total_recall / count,
        "F1 Score": total_f1 / count,
    }


def calculate_bleu(generated_captions, reference_captions):
    total_bleu = 0
    count = len(generated_captions)

    for generated in generated_captions:
        best_bleu = 0

        for references in reference_captions:
            reference_tokens_list = [
                nltk.word_tokenize(reference) for reference in references
            ]
            generated_tokens = nltk.word_tokenize(generated)

            # Calculate BLEU score for each reference
            bleu_score = sentence_bleu(reference_tokens_list, generated_tokens)
            best_bleu = max(best_bleu, bleu_score)

        total_bleu += best_bleu

    return total_bleu / count if count > 0 else 0


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


if __name__ == "__main__":
    # Define paths and device
    model_save_dir = "model"
    validation_data_file = "coco_dataset/annotations/captions_val2014.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    with open(validation_data_file, "r") as f:
        validation_data = json.load(f)

    # Load the model, feature extractor, and tokenizer
    model, feature_extractor, tokenizer = load_model(model_save_dir, device)

    # Initialize lists for generated and reference captions
    generated_captions = []
    reference_captions = {}

    # Counter to track how many images have been processed
    counter = 0
    stop_image_id = 0

    # Process each image in validation data
    for annotation in validation_data["annotations"]:
        image_id = annotation["image_id"]
        reference_caption = annotation["caption"]

        # Collect reference captions for the same image_id
        if image_id not in reference_captions:
            reference_captions[image_id] = []
        reference_captions[image_id].append(reference_caption)

        # Define the path for the image
        image_path = (
            f"coco_dataset/images/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg"
        )

        # Process the image
        image_tensor = process_image(image_path, feature_extractor, device)

        # Generate caption
        caption = generate_caption(model, image_tensor, tokenizer)
        generated_captions.append(caption)

        # Print the generated caption
        print(f"Image ID: {image_id}, Generated Caption: {caption}")

        # Increment the counter
        counter += 1

        # Stop processing if 100 images have been processed
        if counter >= 4000 or image_id == stop_image_id:
            break

    # Evaluate the model
    # ngm_scores = calculate_ngram_metrics(
    #     generated_captions, list(reference_captions.values())
    # )
    bleu_score = calculate_bleu(generated_captions, list(reference_captions.values()))
    rouge_scores = calculate_rouge(
        generated_captions, list(reference_captions.values())
    )

    # Print the evaluation scores
    # print("N-Gram Metrics:", ngm_scores)
    print("BLEU Score:", bleu_score)
    print("ROUGE Scores:", rouge_scores)
