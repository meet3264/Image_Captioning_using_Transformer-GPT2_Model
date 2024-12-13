# import json
# import os
# from collections import Counter
# from test import generate_caption, load_model, process_image

# import nltk
# import pandas as pd
# import torch
# from nltk import ngrams
# from nltk.translate.bleu_score import sentence_bleu
# from PIL import Image
# from rouge import Rouge

# nltk.download("punkt")


# def calculate_ngram_metrics(generated_captions, reference_captions, n=2):
#     total_precision = 0
#     total_recall = 0
#     total_f1 = 0
#     count = 0

#     for generated in generated_captions:
#         best_precision = 0
#         best_recall = 0
#         best_f1 = 0

#         for references in reference_captions:
#             generated_tokens = nltk.word_tokenize(generated)
#             generated_ngrams = Counter(ngrams(generated_tokens, n))
#             overlap = 0

#             for reference in references:
#                 reference_tokens = nltk.word_tokenize(reference)
#                 reference_ngrams = Counter(ngrams(reference_tokens, n))
#                 overlap += sum((generated_ngrams & reference_ngrams).values())

#                 precision = (
#                     overlap / len(generated_ngrams) if len(generated_ngrams) > 0 else 0
#                 )
#                 recall = (
#                     overlap / len(reference_ngrams) if len(reference_ngrams) > 0 else 0
#                 )

#                 f1 = (
#                     (2 * precision * recall) / (precision + recall)
#                     if (precision + recall) > 0
#                     else 0
#                 )

#                 best_precision = max(best_precision, precision)
#                 best_recall = max(best_recall, recall)
#                 best_f1 = max(best_f1, f1)

#         total_precision += best_precision
#         total_recall += best_recall
#         total_f1 += best_f1
#         count += 1

#     return {
#         "Precision": total_precision / count,
#         "Recall": total_recall / count,
#         "F1 Score": total_f1 / count,
#     }


# def calculate_bleu(generated_captions, reference_captions):
#     total_bleu = 0
#     count = len(generated_captions)

#     for generated in generated_captions:
#         best_bleu = 0

#         for references in reference_captions:
#             reference_tokens_list = [
#                 nltk.word_tokenize(reference) for reference in references
#             ]
#             generated_tokens = nltk.word_tokenize(generated)

#             bleu_score = sentence_bleu(reference_tokens_list, generated_tokens)
#             best_bleu = max(best_bleu, bleu_score)

#         total_bleu += best_bleu

#     return total_bleu / count if count > 0 else 0


# def calculate_rouge(generated_captions, reference_captions):
#     rouge = Rouge()
#     rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

#     for generated in generated_captions:
#         best_scores = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

#         for references in reference_captions:
#             for reference in references:
#                 scores = rouge.get_scores(generated, reference)[0]
#                 best_scores["rouge-1"] = max(
#                     best_scores["rouge-1"], scores["rouge-1"]["f"]
#                 )
#                 best_scores["rouge-2"] = max(
#                     best_scores["rouge-2"], scores["rouge-2"]["f"]
#                 )
#                 best_scores["rouge-l"] = max(
#                     best_scores["rouge-l"], scores["rouge-l"]["f"]
#                 )

#         rouge_scores["rouge1"].append(best_scores["rouge-1"])
#         rouge_scores["rouge2"].append(best_scores["rouge-2"])
#         rouge_scores["rougeL"].append(best_scores["rouge-l"])

#     average_scores = {
#         key: sum(value) / len(value) for key, value in rouge_scores.items()
#     }

#     return average_scores


# if __name__ == "__main__":
#     model_save_dir = "model"
#     validation_data_file = "flickr30k_images/results.csv"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the CSV file
#     validation_data = pd.read_csv(
#         validation_data_file, delimiter="|"
#     )  # Specify delimiter if it's not a comma

#     model, feature_extractor, tokenizer = load_model(model_save_dir, device)

#     generated_captions = []
#     reference_captions = {}

#     counter = 0
#     stop_image_id = 0

#     # Track unique images
#     unique_images = set()

#     for i in validation_data.iterrows():
#         image_id = i[1][0]
#         reference_caption = i[1][2]

#         # Add to the dictionary if not already present
#         if image_id not in reference_captions:
#             reference_captions[image_id] = []

#         reference_captions[image_id].append(reference_caption)

#         # Skip if image_id has already been processed
#         if image_id in unique_images:
#             continue

#         # Process and generate caption for the image
#         image_path = f"flickr30k_images/flickr30k_images/{image_id}"  # Adjusted for Flickr dataset
#         image_tensor = process_image(image_path, feature_extractor, device)
#         caption = generate_caption(model, image_tensor, tokenizer)
#         generated_captions.append(caption)

#         print(f"Image ID: {image_id}, Generated Caption: {caption}")

#         # Add to unique images set
#         unique_images.add(image_id)

#         # Stop if we've processed 1000 unique images
#         if len(unique_images) >= 1000:
#             break

#     bleu_score = calculate_bleu(generated_captions, list(reference_captions.values()))
#     rouge_scores = calculate_rouge(
#         generated_captions, list(reference_captions.values())
#     )

#     print("BLEU Score:", bleu_score)
#     print("ROUGE Scores:", rouge_scores)

import json
import os
from collections import Counter
from test import generate_caption, load_model, process_image

import nltk
import pandas as pd
import torch
from nltk import ngrams
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
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
            generated_tokens = nltk.word_tokenize(generated)
            generated_ngrams = Counter(ngrams(generated_tokens, n))
            overlap = 0

            for reference in references:
                reference_tokens = nltk.word_tokenize(reference)
                reference_ngrams = Counter(ngrams(reference_tokens, n))
                overlap += sum((generated_ngrams & reference_ngrams).values())

                precision = (
                    overlap / len(generated_ngrams) if len(generated_ngrams) > 0 else 0
                )
                recall = (
                    overlap / len(reference_ngrams) if len(reference_ngrams) > 0 else 0
                )

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
    smooth_fn = SmoothingFunction().method1
    bleu_scores = {"BLEU-1": 0, "BLEU-2": 0, "BLEU-3": 0, "BLEU-4": 0}
    count = len(generated_captions)

    for i, generated in enumerate(generated_captions):
        generated_tokens = nltk.word_tokenize(generated)
        # Only use corresponding references
        references = reference_captions[i]
        reference_tokens_list = [nltk.word_tokenize(ref) for ref in references]

        # Calculate cumulative n-gram scores
        bleu_scores["BLEU-1"] += sentence_bleu(
            reference_tokens_list,
            generated_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smooth_fn,
        )
        bleu_scores["BLEU-2"] += sentence_bleu(
            reference_tokens_list,
            generated_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth_fn,
        )
        bleu_scores["BLEU-3"] += sentence_bleu(
            reference_tokens_list,
            generated_tokens,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smooth_fn,
        )
        bleu_scores["BLEU-4"] += sentence_bleu(
            reference_tokens_list,
            generated_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth_fn,
        )

    # Average scores
    for key in bleu_scores:
        bleu_scores[key] /= count

    return bleu_scores


# def calculate_bleu_scores(generated_captions, reference_captions):
#     # Tokenize captions
#     references = [
#         [nltk.word_tokenize(ref) for ref in refs] for refs in reference_captions
#     ]
#     hypotheses = [nltk.word_tokenize(gen) for gen in generated_captions]

#     # Define smoothing function
#     smooth_fn = SmoothingFunction().method1

#     # Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 using corpus_bleu with different n-gram weights
#     bleu_1 = corpus_bleu(
#         references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth_fn
#     )
#     bleu_2 = corpus_bleu(
#         references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn
#     )
#     bleu_3 = corpus_bleu(
#         references,
#         hypotheses,
#         weights=(0.33, 0.33, 0.33, 0),
#         smoothing_function=smooth_fn,
#     )
#     bleu_4 = corpus_bleu(
#         references,
#         hypotheses,
#         weights=(0.25, 0.25, 0.25, 0.25),
#         smoothing_function=smooth_fn,
#     )

#     # Return the scores as a dictionary
#     return {"BLEU-1": bleu_1, "BLEU-2": bleu_2, "BLEU-3": bleu_3, "BLEU-4": bleu_4}


def calculate_meteor(generated_captions, reference_captions):
    total_meteor = 0
    count = len(generated_captions)

    for i, generated in enumerate(generated_captions):
        # Tokenize generated and reference captions
        generated_tokens = nltk.word_tokenize(generated)
        reference_tokens_list = [
            nltk.word_tokenize(ref) for ref in reference_captions[i]
        ]

        # Calculate METEOR score for tokenized data
        meteor = meteor_score(reference_tokens_list, generated_tokens)
        total_meteor += meteor

    return total_meteor / count if count > 0 else 0


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
    model_save_dir = "model"
    validation_data_file = "flickr30k_images/results.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_data = pd.read_csv(validation_data_file, delimiter="|")

    model, feature_extractor, tokenizer = load_model(model_save_dir, device)

    generated_captions = []
    reference_captions = {}

    unique_images = set()

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

        if len(unique_images) >= 1000:
            break

    bleu_scores = calculate_bleu(generated_captions, list(reference_captions.values()))
    # bleu_scores = calculate_bleu_scores(
    #     generated_captions, list(reference_captions.values())
    # )
    meteor_score_avg = calculate_meteor(
        generated_captions, list(reference_captions.values())
    )
    rouge_scores = calculate_rouge(
        generated_captions, list(reference_captions.values())
    )

    print("BLEU Scores:", bleu_scores)
    print("METEOR Score:", meteor_score_avg)
    print("ROUGE Scores:", rouge_scores)
