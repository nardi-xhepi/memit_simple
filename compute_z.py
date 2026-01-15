"""
Calcul du vecteur cible z pour MEMIT.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from . import nethook, repr_tools

from .memit_hparams import MEMITHyperParams
from . import generate
from .paraphrase import generate_paraphrases


# Cache pour les templates de contexte
CONTEXT_TEMPLATES_CACHE = None


def get_context_templates(model, tok, use_dynamic=True):
    """Obtient les templates de contexte."""
    global CONTEXT_TEMPLATES_CACHE
    
    if CONTEXT_TEMPLATES_CACHE is None:
        if use_dynamic:
            CONTEXT_TEMPLATES_CACHE = generate.get_context_templates(model, tok)
        else:
            CONTEXT_TEMPLATES_CACHE = [
                ["{}"],
                [
                    "Selon les informations, {}",
                    "Il est connu que {}",
                    "D'après les sources, {}",
                ],
            ]
            print(f"Using static context templates: {CONTEXT_TEMPLATES_CACHE}")
    
    return CONTEXT_TEMPLATES_CACHE


def compute_z(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[List[str]],
) -> torch.Tensor:
    """Calcule le vecteur cible z pour la dernière couche."""
    print(f"Computing target vector (z) at layer {layer}...")
    device = next(model.parameters()).device
    
    # Déterminer la dimension cachée
    if hasattr(model.config, 'text_config'):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    else:
        hidden_size = 3072  # Fallback pour Ministral 3B
    
    # Obtenir les paramètres du LM head
    try:
        lm_w = nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T
    except LookupError:
        # Certains modèles n'ont pas de lm_head séparé
        lm_w = None
    
    try:
        ln_f = nethook.get_module(model, hparams.ln_f_module)
    except LookupError:
        ln_f = None
    
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        # Obtenir vocab_size selon l'architecture
        if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
            vocab_size = model.config.text_config.vocab_size
        elif hasattr(model.config, 'vocab_size'):
            vocab_size = model.config.vocab_size
        else:
            vocab_size = 131072  # Fallback pour Ministral 3B
        lm_b = torch.zeros(vocab_size, device=device)

    # Tokenize la cible
    target_str = request["target_new"]["str"]
    if target_str[0] != " ":
        target_str = " " + target_str
    target_ids = tok(target_str, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"][0]
    print(f"  Target: '{target_str}' -> {target_ids.tolist()}")

    # Générer des paraphrases du prompt original
    print(f"  Generating paraphrases for better generalization...")
    prompt_variations = generate_paraphrases(
        model, tok, request["prompt"], request["subject"], n_paraphrases=3
    )
    
    # Construire les prompts avec toutes les variations
    rewriting_prompts = []
    for prompt_var in prompt_variations:
        for context_types in context_templates:
            for context in context_types:
                rewriting_prompts.append(
                    context.format(prompt_var) + tok.decode(target_ids[:-1])
                )
    
    print(f"  Total rewriting prompts: {len(rewriting_prompts)}")
    kl_prompts = ["{} est"]
    all_prompts = rewriting_prompts + kl_prompts

    # Tokenization
    tok.padding_side = "right"
    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Cibles de réécriture
    rewriting_targets = torch.full(
        (len(rewriting_prompts), input_tok["input_ids"].shape[1]),
        -100,
        device=device,
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Indices de lookup - use subject position like original MEMIT
    # Since we compute loss from traced hidden states, gradients flow correctly
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Couche de loss
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"  Rewrite layer: {layer}")
    print(f"  Loss layer: {loss_layer}")

    # Delta à optimiser - must match model dtype for gradient flow
    model_dtype = next(model.parameters()).dtype
    delta = torch.zeros((hidden_size,), requires_grad=True, device=device, dtype=model_dtype)
    target_init, kl_distr_init = None, None

    rewrite_layer = layer
    layer_module_name = hparams.layer_module_tmp.format(rewrite_layer)

    def edit_output_fn(output, layer):
        nonlocal target_init

        if layer == layer_module_name:
            cur_out = output[0] if isinstance(output, tuple) else output
            
            if target_init is None:
                print("  Recording initial value of z*")
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            # Clone to avoid in-place modification issues with autograd
            cur_out = cur_out.clone()
            for i, idx in enumerate(lookup_idxs):
                if i < cur_out.shape[0]:
                    cur_out[i, idx, :] = cur_out[i, idx, :] + delta

            if isinstance(output, tuple):
                return (cur_out,) + output[1:]
            return cur_out

        return output

    # Optimiseur
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    print(f"  Starting optimization ({hparams.v_num_grad_steps} steps)...")
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(rewrite_layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            model(**input_tok)
            
            # Get hidden states from loss layer (this preserves the gradient graph!)
            loss_layer_output = tr[hparams.layer_module_tmp.format(loss_layer)].output
            if isinstance(loss_layer_output, tuple):
                full_repr = loss_layer_output[0]
            else:
                full_repr = loss_layer_output
            
            # Compute logits directly from hidden states: ln_f(hidden) @ lm_w + lm_b
            # This is the key difference from before - gradients flow through this computation
            if ln_f is not None and lm_w is not None:
                log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
            else:
                # Fallback: use model's lm_head directly
                lm_head = nethook.get_module(model, hparams.lm_head_module)
                logits = lm_head(ln_f(full_repr) if ln_f else full_repr)
                log_probs = torch.log_softmax(logits, dim=2)
            
            # Distribution KL at subject positions
            kl_logits = torch.stack(
                [
                    log_probs[i - len(kl_prompts), idx, :].exp()  # Convert back to probs for extraction
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )
            kl_log_probs = torch.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
            
            # Compute loss on rewriting targets (inside TraceDict for gradient flow)
            log_probs_rewrite = log_probs[:len(rewriting_prompts)]
            loss_per_token = torch.gather(
                log_probs_rewrite,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            nll_loss_each = -(loss_per_token * mask).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * F.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / (torch.norm(target_init) ** 2 + 1e-8)
            )
            loss = nll_loss + kl_loss + weight_decay

        if it % 5 == 0 or it == hparams.v_num_grad_steps - 1:
            prob = torch.exp(-nll_loss_each).mean().item()
            print(
                f"    Step {it:3d}: loss={loss.item():.4f} "
                f"(nll={nll_loss.item():.4f}, kl={kl_loss.item():.4f}) "
                f"P[target]={prob:.4f}"
            )

        if loss < 5e-2:
            print(f"    Converged at step {it}!")
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        loss.backward()
        
        # Debug: check if gradient is flowing
        if it == 0:
            if delta.grad is None:
                print(f"  ⚠️  WARNING: delta.grad is None! Gradient not flowing.")
            else:
                print(f"  ✓ delta.grad norm: {delta.grad.norm().item():.4f}")
        
        opt.step()

        # Projection
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(f"  Init norm: {target_init.norm().item():.4f}")
    print(f"  Delta norm: {delta.norm().item():.4f}")
    print(f"  Target norm: {target.norm().item():.4f}")

    return target.float()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: PreTrainedTokenizer,
    fact_token_strategy: str,
    verbose: bool = True,
) -> int:
    """Trouve l'indice du token de lookup."""
    sentence = prompt.format(subject)
    input_ids = tok(sentence, add_special_tokens=False)["input_ids"]
    seq_len = len(input_ids)
    
    if fact_token_strategy == "last":
        ret = -1
    elif "subject_" in fact_token_strategy:
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
        ret = min(ret, seq_len - 1)
        ret = max(ret, 0)
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    if verbose:
        display_idx = ret if ret >= 0 else seq_len + ret
        token_at_idx = tok.decode([input_ids[display_idx]])
        print(f"  Lookup index: {ret} | Token: '{token_at_idx}'")

    return ret

