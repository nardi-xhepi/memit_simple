"""
MEMIT - Mass-Editing Memory in a Transformer.
https://arxiv.org/abs/2210.07229
"""

from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pathlib import Path
from . import nethook
from .layer_stats import get_inv_cov, compute_covariance

from .compute_z import compute_z, find_fact_lookup_idx
from .compute_ks import compute_ks, get_module_input_at_words
from .memit_hparams import MEMITHyperParams
from . import generate


# Cache global
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
) -> Tuple[PreTrainedModel, Dict[str, torch.Tensor]]:
    """Applique MEMIT au modèle."""
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    device = next(model.parameters()).device

    deltas = execute_memit(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat = key_mat.to(device)
            val_mat = val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.to(w.dtype)

    print(f"✓ Weights updated in {list(deltas.keys())}")

    return model, weights_copy


def execute_memit(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Exécute l'algorithme MEMIT."""
    device = next(model.parameters()).device
    deltas = {}

    # Préparer les requêtes
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    
    for request in requests[:3]:
        print(
            f"MEMIT request: [{request['prompt'].format(request['subject'])}] "
            f"-> [{request['target_new']['str']}]"
        )

    # Récupérer les poids
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = generate.get_context_templates(model, tok)
    print(f"Using {sum(len(ct) for ct in context_templates)} dynamic context templates")

    # Étape 1: Calculer z pour la dernière couche
    z_layer = hparams.layers[-1]
    print(f"\n{'='*60}")
    print(f"Computing target vectors (z) at final layer {z_layer}")
    print(f"{'='*60}")
    
    z_list = []
    for request in requests:
        cur_z = compute_z(
            model, tok, request, hparams, z_layer, context_templates
        )
        z_list.append(cur_z)
    
    zs = torch.stack(z_list, dim=1)  # [hidden_size, num_requests]
    print(f"Target vectors shape: {zs.shape}")

    # Étape 2: Pour chaque couche, calculer les mises à jour
    for i, layer in enumerate(hparams.layers):
        print(f"\n{'='*60}")
        print(f"Processing layer {layer} ({i+1}/{len(hparams.layers)})")
        print(f"{'='*60}")

        # Calculer les clés k pour cette couche
        layer_ks = compute_ks(
            model, tok, requests, hparams, layer, context_templates
        ).T.to(device)  # [hidden_size, num_requests]
        print(f"  Keys shape: {layer_ks.shape}")

        # Calculer les représentations actuelles à z_layer
        cur_zs = get_current_representations(
            model, tok, requests, hparams, z_layer, context_templates
        ).T.to(device)  # [hidden_size, num_requests]
        
        # Erreur résiduelle
        targets = zs.to(device) - cur_zs
        z_error = torch.linalg.norm(targets, dim=0).mean()
        print(f"  Z error: {z_error.item():.4f}")

        # Aligner les dimensions (layer_ks peut avoir plus de colonnes avec plusieurs templates)
        if layer_ks.size(1) != targets.size(1):
            repeat_factor = layer_ks.size(1) // targets.size(1)
            targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Charger/calculer la matrice de covariance
        cov = get_cov(
            model, tok, hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset, hparams.mom2_n_samples,
            hparams.stats_dir, device
        )
        print(f"  Covariance shape: {cov.shape}")

        # Résoudre le système linéaire en double précision
        layer_ks = layer_ks.double()
        targets = targets.double()
        cov = cov.double()


        reg_matrix = hparams.mom2_update_weight * cov + layer_ks @ layer_ks.T
        adj_k = torch.linalg.solve(reg_matrix, layer_ks)
        
        # Distribuer le résidu entre les couches restantes
        remaining_layers = len(hparams.layers) - i
        resid = targets / remaining_layers
        
        # Matrice de mise à jour
        upd_matrix = resid @ adj_k.T
        
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print(f"  Original weight norm: {torch.linalg.norm(weights[weight_name]).item():.4f}")
        print(f"  Update norm: {torch.linalg.norm(upd_matrix).item():.4f}")

        # Appliquer temporairement la mise à jour
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.to(weights[weight_name].dtype)
            deltas[weight_name] = (
                adj_k.detach().cpu().float(),
                resid.detach().cpu().float(),
            )

        # Nettoyer la mémoire
        del cov, layer_ks, targets, adj_k, resid, upd_matrix
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Restaurer les poids originaux
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"\n✓ Deltas computed for {list(weights.keys())}")
    return deltas


def get_current_representations(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[List[str]],
) -> torch.Tensor:
    """Récupère les représentations actuelles à la couche spécifiée."""
    device = next(model.parameters()).device
    
    all_contexts = []
    all_words = []
    
    for request in requests:
        all_contexts.append(request["prompt"])
        all_words.append(request["subject"])
    
    module_name = hparams.layer_module_tmp.format(layer)
    results = []
    
    for context, word in zip(all_contexts, all_words):
        full_text = context.format(word)
        inputs = tok(full_text, return_tensors="pt").to(device)
        
        # Trouver l'indice du sujet
        idx = find_fact_lookup_idx(context, word, tok, hparams.fact_token, verbose=False)
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=False,
                retain_output=True,
            ) as tr:
                model(**inputs)
            
            output = tr.output
            if isinstance(output, tuple):
                output = output[0]
            
            # Gérer l'indice
            seq_len = output.shape[1]
            if idx < 0:
                idx = seq_len + idx
            idx = min(idx, seq_len - 1)
            
            results.append(output[0, idx].float())
    
    return torch.stack(results, dim=0)


def get_cov(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    layer_name: str,
    dataset_name: str,
    n_samples: int,
    stats_dir: str,
    device: torch.device,
) -> torch.Tensor:
    """Récupère la matrice de covariance (pas l'inverse pour MEMIT)."""
    global COV_CACHE
    
    model_name = model.config._name_or_path.replace("/", "_")
    cache_key = f"{model_name}_{layer_name}_{dataset_name}_{n_samples}"
    
    # 1. Cache mémoire
    if cache_key in COV_CACHE:
        print(f"  Using memory-cached covariance")
        return COV_CACHE[cache_key].to(device)
    
    # 2. Cache disque
    if stats_dir:
        cache_path = Path(stats_dir) / f"{cache_key}_cov.pt"
        if cache_path.exists():
            print(f"  Loading cached covariance from {cache_path}")
            cov = torch.load(cache_path, weights_only=True)
            COV_CACHE[cache_key] = cov
            return cov.to(device)
    
    # 3. Calculer la covariance
    print(f"  Computing covariance for {layer_name}...")
    cov = compute_covariance(
        model, tok, layer_name, dataset_name, n_samples, "float32"
    )
    
    # Sauvegarder en mémoire
    COV_CACHE[cache_key] = cov.cpu()
    
    # Sauvegarder sur disque
    if stats_dir:
        cache_path = Path(stats_dir) / f"{cache_key}_cov.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cov.cpu(), cache_path)
        print(f"  Saved covariance to {cache_path}")
    
    return cov.to(device)


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Adapte la forme de la matrice de mise à jour."""
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            f"Update matrix shape {matrix.shape} doesn't match weight shape {shape}"
        )


def get_context_templates_local() -> List[List[str]]:
    """Retourne les templates de contexte."""
    global CONTEXT_TEMPLATES_CACHE
    
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [
            ["{}"],  # Direct
            [
                "Selon les informations, {}",
                "Il est connu que {}",
                "D'après les sources, {}",
            ],
        ]
        print(f"Using {sum(len(ct) for ct in CONTEXT_TEMPLATES_CACHE)} context templates")
    
    return CONTEXT_TEMPLATES_CACHE


def reset_context_templates():
    """Réinitialise le cache des templates."""
    global CONTEXT_TEMPLATES_CACHE
    CONTEXT_TEMPLATES_CACHE = None

