#!/usr/bin/env python3
"""
Weighted ensemble inference for spam classification.
"""

import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

from autograder.dataset import CPEN455_2025_W1_Dataset
from model import LlamaModel
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from examples.bayes_inverse import bayes_inverse_llm_classifier


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


def weighted_ensemble_inference(args, models, weights, tokenizer, dataloader, device):
    """Run inference with multiple models and compute weighted average predictions."""
    all_predictions = {}
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()
    print(f"Normalized weights: {weights.tolist()}")
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Weighted ensemble inference"):
            data_indices, subjects, messages, labels = batch
            indices = torch.as_tensor(data_indices).view(-1).tolist()
            
            batch_probs = []
            for model in models:
                _, (probs, _) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device)
                batch_probs.append(probs)
            
            stacked = torch.stack(batch_probs, dim=0)
            weighted_probs = (stacked * weights.view(-1, 1, 1)).sum(dim=0)
            
            for i, idx in enumerate(indices):
                all_predictions[idx] = {
                    "prob_ham": weighted_probs[i, 0].item(),
                    "prob_spam": weighted_probs[i, 1].item(),
                }
    
    return all_predictions


def save_predictions(predictions, output_path):
    """Save predictions to CSV."""
    rows = []
    for idx in sorted(predictions.keys()):
        rows.append({
            "data_index": idx,
            "prob_ham": predictions[idx]["prob_ham"],
            "prob_spam": predictions[idx]["prob_spam"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved weighted ensemble predictions to {output_path}")
    
    df["pred"] = (df["prob_spam"] > df["prob_ham"]).astype(int)
    spam_count = df["pred"].sum()
    ham_count = len(df) - spam_count
    print(f"Predictions: {spam_count} spam, {ham_count} ham")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoints for ensemble")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Weights for each model (default: equal weights)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--test_dataset_path", type=str, 
                        default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--output", type=str, 
                        default="bayes_inverse_probs/test_dataset_probs.csv")
    parser.add_argument("--user_prompt", type=str, default="")
    args = parser.parse_args()
    
    load_dotenv()
    
    base_checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    device = set_device()
    
    tokenizer = Tokenizer.from_pretrained(base_checkpoint, cache_dir=model_cache_dir)
    base_path = _resolve_snapshot_path(base_checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)
    
    print(f"Loading {len(args.checkpoints)} models for weighted ensemble...")
    models = []
    for ckpt_path in args.checkpoints:
        model = load_model_from_checkpoint(
            ckpt_path, config, base_checkpoint, model_cache_dir, device
        )
        models.append(model)
    
    if args.weights is None:
        weights = [1.0] * len(models)
    else:
        assert len(args.weights) == len(models), "Number of weights must match number of models"
        weights = args.weights
    
    print(f"Using weights: {weights}")
    
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    predictions = weighted_ensemble_inference(args, models, weights, tokenizer, test_dataloader, device)
    save_predictions(predictions, args.output)
