"""
Causal Tracing for MEMIT - Ministral 3B Implementation.

This module implements the causal tracing methodology from the MEMIT paper
to identify which layers store factual knowledge. Adapted for Ministral 3B
multimodal architecture.

Reference: Meng et al., "Mass-Editing Memory in a Transformer" (2022)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import PreTrainedModel, PreTrainedTokenizer

from . import nethook


@dataclass
class CausalTraceConfig:
    """Configuration for causal tracing experiments."""
    
    # Noise level (multiplier for embedding std, or absolute value)
    noise_level: float = 3.0
    
    # Number of corrupted samples for averaging
    samples: int = 10
    
    # Window size for MLP/Attn ablation studies
    window: int = 10
    
    # Whether to use uniform noise (vs Gaussian)
    uniform_noise: bool = False
    
    # Whether to replace embeddings (vs add noise)
    replace: bool = False
    
    # Number of layers in the model (auto-detected if None)
    num_layers: Optional[int] = None


def layername(model: PreTrainedModel, num: int, kind: Optional[str] = None) -> str:
    """
    Get the module name for a specific layer in Ministral 3B.
    
    Args:
        model: The model (used for architecture detection)
        num: Layer number
        kind: Component type - None (full layer), "mlp", "attn", or "embed"
        
    Returns:
        Full module path string
    """
    # Ministral 3B multimodal architecture
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        # Mistral3ForConditionalGeneration structure
        if kind == "embed":
            return "model.language_model.embed_tokens"
        base = f"model.language_model.layers.{num}"
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Standard MistralForCausalLM structure
        if kind == "embed":
            return "model.embed_tokens"
        base = f"model.layers.{num}"
    else:
        # Fallback for other architectures
        if kind == "embed":
            return "model.embed_tokens"
        base = f"model.layers.{num}"
    
    if kind is None:
        return base
    elif kind == "mlp":
        return f"{base}.mlp"
    elif kind == "attn":
        return f"{base}.self_attn"
    
    return base


def get_num_layers(model: PreTrainedModel) -> int:
    """Detect the number of transformer layers in the model."""
    if hasattr(model.config, 'text_config'):
        return model.config.text_config.num_hidden_layers
    elif hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    else:
        # Fallback: count layers
        return 26  # Ministral 3B default


def make_inputs(
    tok: PreTrainedTokenizer,
    prompts: List[str],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create batched inputs with left-padding for causal models.
    
    Args:
        tok: Tokenizer
        prompts: List of prompt strings
        device: Target device
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    token_lists = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    
    # Use pad token or fallback to 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    
    # Left-padding for causal LM
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    
    return {
        "input_ids": torch.tensor(input_ids).to(device),
        "attention_mask": torch.tensor(attention_mask).to(device),
    }


def decode_tokens(tok: PreTrainedTokenizer, token_array) -> List[str]:
    """Decode tokens one by one for visualization."""
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tok, row) for row in token_array]
    return [tok.decode([t]) for t in token_array]


def find_token_range(
    tok: PreTrainedTokenizer,
    token_array: torch.Tensor,
    substring: str,
) -> Tuple[int, int]:
    """
    Find the token range corresponding to a substring.
    
    Args:
        tok: Tokenizer
        token_array: 1D tensor of token IDs
        substring: The substring to locate
        
    Returns:
        Tuple of (start_idx, end_idx) token positions
    """
    toks = decode_tokens(tok, token_array)
    whole_string = "".join(toks)
    
    # Try to find substring, with fallback for leading space
    try:
        char_loc = whole_string.index(substring)
    except ValueError:
        try:
            char_loc = whole_string.index(f" {substring}")
        except ValueError:
            # Default to first token if not found
            print(f"  Warning: '{substring}' not found in '{whole_string}'")
            return (0, 1)
    
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    
    return (tok_start or 0, tok_end or len(toks))


def predict_from_input(
    model: PreTrainedModel,
    inp: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get model predictions and probabilities.
    
    Returns:
        Tuple of (predicted_token_ids, probabilities)
    """
    with torch.no_grad():
        out = model(**inp).logits
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def collect_embedding_std(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    subjects: List[str],
) -> float:
    """
    Compute the standard deviation of subject embeddings.
    Used to calibrate noise level for causal tracing.
    
    Args:
        model: The model
        tok: Tokenizer
        subjects: List of subject strings
        
    Returns:
        Standard deviation of embeddings
    """
    device = next(model.parameters()).device
    embed_layer = layername(model, 0, "embed")
    
    alldata = []
    for s in subjects:
        inp = make_inputs(tok, [s], device=device)
        with nethook.Trace(model, embed_layer) as t:
            model(**inp)
            alldata.append(t.output[0])
    
    alldata = torch.cat(alldata)
    return alldata.std().item()


def trace_with_patch(
    model: PreTrainedModel,
    inp: Dict[str, torch.Tensor],
    states_to_patch: List[Tuple[int, str]],
    answers_t: int,
    tokens_to_mix: Tuple[int, int],
    noise: float = 0.1,
    uniform_noise: bool = False,
    replace: bool = False,
    trace_layers: Optional[List[str]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Core causal tracing function.
    
    Runs inference with:
    - inp[0]: Clean run (uncorrupted)
    - inp[1:]: Corrupted runs (noise added to subject tokens)
    - states_to_patch: List of (token_idx, layer_name) to restore from clean run
    
    Args:
        model: The model
        inp: Batched inputs (batch_size >= 2)
        states_to_patch: List of (token_index, layer_name) to restore
        answers_t: Target answer token ID
        tokens_to_mix: (begin, end) range of tokens to corrupt
        noise: Noise level
        uniform_noise: Use uniform instead of Gaussian noise
        replace: Replace embeddings instead of adding noise
        trace_layers: Optional list of layers to trace outputs
        
    Returns:
        Probability of target answer (and optionally traced activations)
    """
    rs = np.random.RandomState(1)  # Reproducible noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    
    # Group patches by layer
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    
    embed_layername = layername(model, 0, "embed")
    
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    
    # Define noise function
    if isinstance(noise, (int, float)):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise
    
    def patch_rep(x, layer):
        """Hook function to corrupt/restore representations."""
        if layer == embed_layername:
            # Corrupt subject token embeddings for batch items [1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device).to(x.dtype)
                
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        
        if layer not in patch_spec:
            return x
        
        # Restore from clean run (batch item 0)
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x
    
    # Run with patching
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)
    
    # Get probability for target answer
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    
    # Optionally return traced activations
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers],
            dim=2,
        )
        return probs, all_traced
    
    return probs


def trace_important_states(
    model: PreTrainedModel,
    num_layers: int,
    inp: Dict[str, torch.Tensor],
    e_range: Tuple[int, int],
    answer_t: int,
    noise: float = 0.1,
    uniform_noise: bool = False,
    replace: bool = False,
    token_range: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Trace importance of each (token, layer) state.
    
    Returns:
        2D tensor of shape [num_tokens, num_layers] with recovery scores
    """
    ntoks = inp["input_ids"].shape[1]
    table = []
    
    if token_range is None:
        token_range = range(ntoks)
    
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    
    return torch.stack(table)


def trace_important_window(
    model: PreTrainedModel,
    num_layers: int,
    inp: Dict[str, torch.Tensor],
    e_range: Tuple[int, int],
    answer_t: int,
    kind: str,
    window: int = 10,
    noise: float = 0.1,
    uniform_noise: bool = False,
    replace: bool = False,
    token_range: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Trace importance with a sliding window of layers (for MLP/Attn ablation).
    """
    ntoks = inp["input_ids"].shape[1]
    table = []
    
    if token_range is None:
        token_range = range(ntoks)
    
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2),
                    min(num_layers, layer + (window + 1) // 2),
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    
    return torch.stack(table)


def calculate_hidden_flow(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    prompt: str,
    subject: str,
    config: Optional[CausalTraceConfig] = None,
    expect: Optional[str] = None,
    kind: Optional[str] = None,
) -> Dict:
    """
    Main API: Run causal tracing over all tokens and layers.
    
    Args:
        model: The model
        tok: Tokenizer
        prompt: The prompt string (subject should appear in it)
        subject: The subject to corrupt
        config: Tracing configuration
        expect: Expected answer (skip if prediction doesn't match)
        kind: None for full layer, "mlp" or "attn" for components
        
    Returns:
        Dictionary with:
        - scores: 2D tensor [num_tokens, num_layers]
        - low_score: Corrupted baseline probability
        - high_score: Clean baseline probability
        - input_tokens: List of decoded tokens
        - subject_range: (start, end) token indices
        - answer: Predicted answer token
        - correct_prediction: Boolean
    """
    if config is None:
        config = CausalTraceConfig()
    
    device = next(model.parameters()).device
    num_layers = config.num_layers or get_num_layers(model)
    
    # Create batch: [clean] + [corrupted] * samples
    inp = make_inputs(tok, [prompt] * (config.samples + 1), device=device)
    
    # Get baseline prediction
    preds, probs = predict_from_input(model, inp)
    answer_t = preds[0].item()
    base_score = probs[0].item()
    answer = tok.decode([answer_t])
    
    # Check expected answer
    if expect is not None and answer.strip() != expect.strip():
        return dict(correct_prediction=False, answer=answer)
    
    # Find subject token range
    e_range = find_token_range(tok, inp["input_ids"][0], subject)
    
    # Get corrupted baseline
    low_score = trace_with_patch(
        model,
        inp,
        [],  # No restoration
        answer_t,
        e_range,
        noise=config.noise_level,
        uniform_noise=config.uniform_noise,
    ).item()
    
    # Trace all (token, layer) combinations
    if kind is None:
        differences = trace_important_states(
            model,
            num_layers,
            inp,
            e_range,
            answer_t,
            noise=config.noise_level,
            uniform_noise=config.uniform_noise,
            replace=config.replace,
        )
    else:
        differences = trace_important_window(
            model,
            num_layers,
            inp,
            e_range,
            answer_t,
            kind=kind,
            window=config.window,
            noise=config.noise_level,
            uniform_noise=config.uniform_noise,
            replace=config.replace,
        )
    
    return dict(
        scores=differences.detach().cpu(),
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0].cpu(),
        input_tokens=decode_tokens(tok, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=config.window,
        correct_prediction=True,
        kind=kind or "",
    )


def find_important_layers(
    result: Dict,
    n_top: int = 5,
    token_position: str = "last",
) -> List[Tuple[int, float]]:
    """
    Analyze causal trace results to find the most important layers.
    
    Args:
        result: Output from calculate_hidden_flow
        n_top: Number of top layers to return
        token_position: "last" for last token, "subject" for subject tokens
        
    Returns:
        List of (layer_idx, score) tuples, sorted by importance
    """
    scores = result["scores"]
    
    if token_position == "last":
        # Use last token position
        layer_scores = scores[-1].numpy()
    elif token_position == "subject":
        # Average over subject token positions
        start, end = result["subject_range"]
        layer_scores = scores[start:end].mean(dim=0).numpy()
    else:
        # Average over all tokens
        layer_scores = scores.mean(dim=0).numpy()
    
    # Rank by score
    ranked = sorted(enumerate(layer_scores), key=lambda x: x[1], reverse=True)
    return ranked[:n_top]


def suggest_memit_layers(
    result: Dict,
    n_layers: int = 4,
) -> List[int]:
    """
    Suggest layers to target for MEMIT based on causal trace.
    
    Returns a contiguous range of layers centered on the most important ones.
    
    Args:
        result: Output from calculate_hidden_flow
        n_layers: Number of layers to suggest
        
    Returns:
        List of layer indices
    """
    important = find_important_layers(result, n_top=n_layers * 2, token_position="last")
    
    if not important:
        # Fallback to middle layers
        num_layers = result["scores"].shape[1]
        mid = num_layers // 2
        return list(range(mid - n_layers // 2, mid + (n_layers + 1) // 2))
    
    # Find the centroid of top layers
    top_layers = [idx for idx, _ in important[:n_layers]]
    center = int(np.mean(top_layers))
    
    # Create contiguous range around center
    start = center - n_layers // 2
    start = max(0, start)
    
    return list(range(start, start + n_layers))


def plot_trace_heatmap(
    result: Dict,
    savepdf: Optional[str] = None,
    title: Optional[str] = None,
    modelname: str = "Ministral",
) -> None:
    """
    Plot a heatmap visualization of causal trace results.
    
    Args:
        result: Output from calculate_hidden_flow
        savepdf: Optional path to save PDF
        title: Optional custom title
        modelname: Model name for axis label
    """
    import os
    
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result.get("answer", "")
    kind = result.get("kind") or None
    labels = list(result["input_tokens"])
    
    # Mark subject tokens
    start, end = result["subject_range"]
    for i in range(start, end):
        if i < len(labels):
            labels[i] = labels[i] + "*"
    
    # Color maps by component type
    cmap = {
        None: "Purples",
        "": "Purples",
        "mlp": "Greens",
        "attn": "Reds",
    }.get(kind, "Purples")
    
    fig, ax = plt.subplots(figsize=(max(5, differences.shape[1] * 0.3), 3), dpi=150)
    
    h = ax.pcolor(differences.numpy(), cmap=cmap, vmin=low_score)
    ax.invert_yaxis()
    
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1], 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1], 5)))
    
    if title:
        ax.set_title(title, fontsize=10)
    elif kind:
        kindname = "MLP" if kind == "mlp" else "Attn"
        ax.set_title(f"Impact of restoring {kindname} after corruption", fontsize=10)
    else:
        ax.set_title("Impact of restoring state after corruption", fontsize=10)
    
    ax.set_xlabel(f"Layer within {modelname}", fontsize=9)
    
    cb = plt.colorbar(h, ax=ax, shrink=0.8)
    if answer:
        cb.ax.set_title(f"P({answer.strip()})", fontsize=9, pad=5)
    
    plt.tight_layout()
    
    if savepdf:
        os.makedirs(os.path.dirname(savepdf) or ".", exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap to {savepdf}")
    else:
        plt.show()


def run_causal_trace(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    prompt: str,
    subject: str,
    expected_answer: Optional[str] = None,
    config: Optional[CausalTraceConfig] = None,
    plot: bool = True,
    savepdf: Optional[str] = None,
) -> Dict:
    """
    Convenience function to run causal tracing and optionally plot results.
    
    Args:
        model: The model
        tok: Tokenizer
        prompt: Prompt string
        subject: Subject to corrupt
        expected_answer: Expected answer (for verification)
        config: Tracing configuration
        plot: Whether to plot the heatmap
        savepdf: Optional path to save PDF
        
    Returns:
        Causal trace results dictionary
    """
    print(f"Running causal trace...")
    print(f"  Prompt: {prompt}")
    print(f"  Subject: {subject}")
    
    result = calculate_hidden_flow(
        model, tok, prompt, subject, config=config, expect=expected_answer
    )
    
    if not result.get("correct_prediction", True):
        print(f"  Warning: Model predicted '{result.get('answer')}' instead of '{expected_answer}'")
    else:
        print(f"  Answer: {result['answer']}")
        print(f"  High score (clean): {result['high_score']:.4f}")
        print(f"  Low score (corrupt): {result['low_score']:.4f}")
        
        # Show important layers
        important = find_important_layers(result, n_top=5)
        print(f"  Top 5 layers:")
        for layer, score in important:
            print(f"    Layer {layer}: {score:.4f}")
        
        suggested = suggest_memit_layers(result)
        print(f"  Suggested MEMIT layers: {suggested}")
    
    if plot and result.get("correct_prediction", True):
        plot_trace_heatmap(result, savepdf=savepdf)
    
    return result
