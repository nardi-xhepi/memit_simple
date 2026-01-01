"""
Calcul des statistiques de couche (covariance).
"""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

from . import nethook


COV_CACHE: Dict[str, torch.Tensor] = {}


def get_inv_cov(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    layer_name: str,
    dataset_name: str = "wikitext",
    n_samples: int = 1000,
    dtype: str = "float32",
    stats_dir: Optional[str] = None,
) -> torch.Tensor:
    """Calcule ou récupère l'inverse de la matrice de covariance."""
    global COV_CACHE
    

    model_name = model.config._name_or_path.replace("/", "_")
    cache_key = f"{model_name}_{layer_name}_{dataset_name}_{n_samples}"
    
    if cache_key in COV_CACHE:
        print(f"  Using cached covariance for {layer_name}")
        return COV_CACHE[cache_key]
    
    # Vérifier si fichier de cache existe
    if stats_dir:
        cache_path = Path(stats_dir) / f"{cache_key}_inv_cov.pt"
        if cache_path.exists():
            print(f"  Loading cached covariance from {cache_path}")
            inv_cov = torch.load(cache_path, weights_only=True)
            COV_CACHE[cache_key] = inv_cov.to(next(model.parameters()).device)
            return COV_CACHE[cache_key]
    
    print(f"  Computing covariance matrix for {layer_name}...")
    print(f"  Dataset: {dataset_name}, Samples: {n_samples}")
    
    # Calculer la covariance
    cov_matrix = compute_covariance(
        model, tok, layer_name, dataset_name, n_samples, dtype
    )
    
    print("  Inverting covariance matrix...")
    device = cov_matrix.device
    
    reg = 1e-4 * torch.eye(cov_matrix.shape[0], device=device, dtype=cov_matrix.dtype)
    cov_matrix = cov_matrix + reg
    
    try:
        inv_cov = torch.linalg.inv(cov_matrix).float()
    except RuntimeError:
        print("  Warning: Matrix inversion failed, using pseudo-inverse")
        inv_cov = torch.linalg.pinv(cov_matrix).float()
    

    COV_CACHE[cache_key] = inv_cov
    
    if stats_dir:
        cache_path = Path(stats_dir) / f"{cache_key}_inv_cov.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inv_cov.cpu(), cache_path)
        print(f"  Saved covariance to {cache_path}")
    
    return inv_cov


def compute_covariance(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    layer_name: str,
    dataset_name: str,
    n_samples: int,
    dtype: str,
) -> torch.Tensor:
    """Calcule la matrice de covariance des activations. C = E[x @ x.T]"""
    device = next(model.parameters()).device
    torch_dtype = torch.float32 if dtype == "float32" else torch.float16
    
    # Charger le dataset
    print(f"  Loading dataset {dataset_name}...")
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text_column = "text"
    elif dataset_name == "wikipedia":
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        ds = list(ds.take(n_samples * 3))
        text_column = "text"
    elif dataset_name == "wikipedia_fr":
        print("  Loading French Wikipedia (streaming)...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
        ds = list(ds.take(n_samples * 3))  # Prendre plus pour filtrer les courts
        text_column = "text"
    elif dataset_name == "cc100_fr":
        print("  Loading CC100 French...")
        ds = load_dataset("cc100", "fr", split="train", streaming=True)
        ds = list(ds.take(n_samples * 3))
        text_column = "text"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use: wikitext, wikipedia, wikipedia_fr, cc100_fr")
    

    if isinstance(ds, list):
        texts = [item[text_column] for item in ds if len(item.get(text_column, "").strip()) > 100]
    else:
        texts = [t for t in ds[text_column] if len(t.strip()) > 100]
    texts = texts[:n_samples * 2]
    
    sum_xx = None
    n_tokens = 0
    hidden_size = None
    
    model.eval()
    nethook.set_requires_grad(False, model)
    
    batch_size = 4
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    print(f"  Processing {n_samples} samples (texts)...")
    pbar = tqdm(total=min(n_samples, len(texts)), desc="  Computing covariance")
    
    samples_processed = 0
    
    for i in range(0, len(texts), batch_size):
        if samples_processed >= n_samples:
            break
            
        batch_texts = texts[i:i + batch_size]
        

        inputs = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        

        with torch.no_grad():
            with nethook.Trace(model, layer_name, retain_input=True, retain_output=False) as tr:
                try:
                    model(**inputs)
                except Exception as e:
                    continue
                
            acts = tr.input
            if isinstance(acts, tuple):
                acts = acts[0]

            acts = acts.to(torch_dtype)
            
            if hidden_size is None:
                hidden_size = acts.shape[-1]
                sum_xx = torch.zeros((hidden_size, hidden_size), dtype=torch_dtype, device=device)
            
            acts_flat = acts.reshape(-1, hidden_size)
            
            sum_xx += acts_flat.T @ acts_flat
            n_tokens += acts_flat.shape[0]
            samples_processed += len(batch_texts)
            
            pbar.update(len(batch_texts))
    
    pbar.close()
    
    cov_matrix = sum_xx / n_tokens
    
    print(f"  Covariance matrix shape: {cov_matrix.shape}")
    print(f"  Samples processed: {samples_processed}, Total tokens: {n_tokens}")
    
    return cov_matrix.float()

