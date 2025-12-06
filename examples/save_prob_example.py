#!/usr/bin/env python3

"""
Ensemble save probabilities for SmolLM2-135M-Instruct.

Filepath: ./examples/save_prob_example.py
Project: CPEN455-Project-2025W1
Description: This script loads multiple fine-tuned models and averages their predictions
             to produce ensemble probabilities for the test dataset.

Usage:
    uv run -m examples.save_prob_example
"""

import os
from dotenv import load_dotenv
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from autograder.dataset import CPEN455_2025_W1_Dataset
from model import LlamaModel
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from examples.bayes_inverse import bayes_inverse_llm_classifier

CKPTS_DIR = os.path.join(os.path.dirname(__file__), "ckpts")
ENSEMBLE_CHECKPOINTS = [
    os.path.join(CKPTS_DIR, "final_model_seed0.ckpt"),
    os.path.join(CKPTS_DIR, "final_model_seed42.ckpt"),
    os.path.join(CKPTS_DIR, "final_model_seed123.ckpt"),
]


def load_model_from_checkpoint(ckpt_path, config, base_checkpoint, model_cache_dir, device):
    """Load a model from a checkpoint file."""
    model = LlamaModel(config)
    load_model_weights(model, base_checkpoint, cache_dir=model_cache_dir, device=device)
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        seed = ckpt.get("seed", "unknown")
        iteration = ckpt.get("iteration", "unknown")
        print(f"Loaded {ckpt_path} (seed={seed}, iter={iteration})")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    return model.to(device)


def ensemble_save_probs(args, models, tokenizer, dataloader, device, name="test"):
    """Run ensemble inference and save averaged probabilities."""
    save_path = os.path.join(os.getcwd(), f"{args.prob_output_folder}/{name}_dataset_probs.csv")
    
    if os.path.exists(save_path):
        os.remove(save_path)
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ensemble inference"):
            data_indices, _, _, _ = batch
            indices = torch.as_tensor(data_indices).view(-1).tolist()
            
            batch_probs = []
            for model in models:
                _, (probs, _) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device)
                batch_probs.append(probs)
            
            # Average probabilities across models
            stacked = torch.stack(batch_probs, dim=0)
            avg_probs = stacked.mean(dim=0)
            
            rows = zip(indices, avg_probs[:, 0].tolist(), avg_probs[:, 1].tolist())
            
            file_exists = os.path.exists(save_path)
            with open(save_path, "a", newline="") as handle:
                if not file_exists:
                    handle.write("data_index,prob_ham,prob_spam\n")
                handle.writelines(f"{idx},{ham},{spam}\n" for idx, ham, spam in rows)
    
    print(f"Saved ensemble probabilities to {save_path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--test_dataset_path", type=str, default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")
    args = parser.parse_args()

    load_dotenv()
    
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    
    device = set_device()

    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=model_cache_dir)
    
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    # Load ensemble models
    print(f"Loading {len(ENSEMBLE_CHECKPOINTS)} models for ensemble...")
    models = []
    for ckpt_path in ENSEMBLE_CHECKPOINTS:
        model = load_model_from_checkpoint(ckpt_path, config, checkpoint, model_cache_dir, device)
        models.append(model)
    print(f"Loaded {len(models)} models for ensemble inference")
    
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    ensemble_save_probs(args, models, tokenizer, test_dataloader, device=device, name="test")
