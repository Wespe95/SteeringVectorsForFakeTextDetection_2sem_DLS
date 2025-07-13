"""
Sterring Vector Calculator Module
This module contains functions for calculating the Sterring Vector from model activations.
"""

import os
import pickle
import torch
import steering_vectors
import logging
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    layer_type: str,
    device: str
) -> Dict[int, List[torch.Tensor]]:
    """
    Collects activations for a list of texts (processed individually).
    """
    all_layer_activations = {}

    # Process texts one by one
    for text in tqdm(texts, desc="Processing texts"):
        # Tokenize single text
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True
        ).to(device)

        # Ensure all tensors are on the correct device
        #tokenized = {k: v.to(device) for k, v in tokenized.items()}

        # Record activations
        with torch.no_grad():
            with steering_vectors.record_activations(model, layer_type=layer_type) as recorded_activations:
                model(**tokenized)

        # Process recorded activations
        for layer_idx, layer_activations in recorded_activations.items():
            if layer_idx not in all_layer_activations:
                all_layer_activations[layer_idx] = []

            # Should be only one activation since we process one text at a time
            for activation in layer_activations:
                # activation shape: [1, seq_len, hidden_dim]
                # Average over sequence length dimension
                token_averaged = activation.mean(dim=1).squeeze(0)  # [hidden_dim]
                all_layer_activations[layer_idx].append(token_averaged.cpu())

    # Convert lists to tensors
    for layer_idx in all_layer_activations:
        all_layer_activations[layer_idx] = torch.stack(all_layer_activations[layer_idx])

    return all_layer_activations


def calculate_steering_from_activations(
    real_activations: Dict[int, torch.Tensor],
    fake_activations: Dict[int, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Calculates steering vectors from the collected activations.
    """
    steering_vectors = {}

    # Get all layer indices
    layer_indices = sorted(set(real_activations.keys()) & set(fake_activations.keys()))

    # Calculate per-layer steering vectors
    layer_steering_vectors = []

    for layer_idx in layer_indices:
        real_acts = real_activations[layer_idx]
        fake_acts = fake_activations[layer_idx]

        # Calculate mean difference (fake - real)
        real_mean = real_acts.mean(dim=0)
        fake_mean = fake_acts.mean(dim=0)

        layer_steering = fake_mean - real_mean
        layer_steering_vectors.append(layer_steering)

        steering_vectors[f"layer_{layer_idx}"] = layer_steering

    # Calculate overall steering vector (average across all layers)
    if layer_steering_vectors:
        overall_steering = torch.stack(layer_steering_vectors).mean(dim=0)
        steering_vectors["overall"] = overall_steering

    return steering_vectors, overall_steering


def extract_and_organize_data(
    datasets: Dict[str, any],
    real_label: int = 0,
    fake_label: int = 1
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extracts and organizes the data from the dataset objects.

    Returns:
        Organized data in the format {"train": {"real": [...], "fake": [...]}, "dev": {...}}
    """
    organized_data = {}

    for split_name, dataset in datasets.items():
        # Konvertiere zu pandas DataFrame für einfachere Filterung
        df = dataset.to_pandas()

        # Filtere nach Labels
        real_texts = df[df['label'] == real_label]['text'].tolist()
        fake_texts = df[df['label'] == fake_label]['text'].tolist()

        organized_data[split_name] = {
            "real": real_texts,
            "fake": fake_texts
        }

        print(f"{split_name}: {len(real_texts)} real texts, {len(fake_texts)} fake texts")

    return organized_data


def calculate_steering_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    datasets: Dict[str, any],
    real_label: int = 0,
    fake_label: int = 1,
    save_dir: str = ".steering_data/cache",
    max_length: int = 512,
    #batch_size: int = 4,
    layer_type: str = "decoder_block",
    only_dev: bool = False,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict[str, torch.Tensor]:
    """
    Calculates steering vectors for an LLM based on true and false texts.
    """

    model = model.to(device)
    model.eval()

    #if tokenizer.pad_token is None:
    #    if tokenizer.eos_token is not None:
    #        tokenizer.pad_token = tokenizer.eos_token
    #        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    #    else:
    #        # Fallback: add a new pad token
    #        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #        model.resize_token_embeddings(len(tokenizer))
    #        logger.info("Added new [PAD] token and resized model embeddings")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Starting steering vector calculation on {device}")
    organized_data = extract_and_organize_data(datasets, real_label, fake_label)
    logger.info(f"Train data: {len(organized_data['train']['real'])} real, {len(organized_data['train']['fake'])} fake texts")
    logger.info(f"Dev data: {len(organized_data['dev']['real'])} real, {len(organized_data['dev']['fake'])} fake texts")

    # Process data and collect activations
    results = {}

    for data_split, data in tqdm(organized_data.items(), desc="Processing data splits"):
        logger.info(f"Processing {data_split} data...")

        split_activations = {}

        for label, texts in data.items():
            logger.info(f"Processing {len(texts)} {label} texts from {data_split}")

            # Check if activations already exist
            activation_file = os.path.join(save_dir, f"{data_split}_{label}_activations.pkl")

            if os.path.exists(activation_file):
                logger.info(f"Loading existing activations from {activation_file}")
                with open(activation_file, 'rb') as f:
                    label_activations = pickle.load(f)
            else:
                label_activations = collect_activations( #max_length
                    model, tokenizer, texts, max_length,
                    layer_type, device
                )

                # Save activations
                logger.info(f"Saving activations to {activation_file}")
                with open(activation_file, 'wb') as f:
                    pickle.dump(label_activations, f)

            split_activations[label] = label_activations

        results[data_split] = split_activations

    # Calculate steering vectors
    if not only_dev and "train" in results: #Важно для переноса домена, если мы хотим тренировать классификатор на одном множестве, а тестить на немного другом
        logger.info("Calculating steering vectors...")
        steering_vectors, overall_steering = calculate_steering_from_activations(
            results["train"]["real"],
            results["train"]["fake"]
        )

        # Save steering vectors
        steering_file = os.path.join(save_dir, "steering_vectors.pkl")
        logger.info(f"Saving steering vectors to {steering_file}")
        with open(steering_file, 'wb') as f:
            pickle.dump(steering_vectors, f)

        # Save overall_steering
        steering_file = os.path.join(save_dir, "overall_steering_vector.pkl")
        logger.info(f"Saving steering vectors to {steering_file}")
        with open(steering_file, 'wb') as f:
            pickle.dump(overall_steering, f)
    else:
        logger.info("only_dev=True or no train split: Steering vectors are NOT calculated")
        steering_vectors = None
        overall_steering = None

    # Save complete results
    final_results = {
        "steering_vectors": steering_vectors,
        "overall_steering": overall_steering,
        "metadata": {
            "train_real_count": len(organized_data["train"]["real"]) if "train" in organized_data else 0,
            "train_fake_count": len(organized_data["train"]["fake"]) if "train" in organized_data else 0,
            "dev_real_count": len(organized_data["dev"]["real"]),
            "dev_fake_count": len(organized_data["dev"]["fake"]),
            "max_length": max_length,
            "layer_type": layer_type,
            "device": device,
            "real_label": real_label,
            "fake_label": fake_label
        }
    }

    results_file = os.path.join(save_dir, "complete_results.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(final_results, f)

    logger.info("Steering vector calculation completed!")
    return final_results

