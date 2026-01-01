"""
Génération de texte pour créer des templates de contexte dynamiques.
Support du français et de l'anglais.
Utilise le device du modèle en entrée.
"""

import unicodedata
from typing import List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


# Cache pour les templates générés
CONTEXT_TEMPLATES_CACHE = None


def get_context_templates(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    language: str = "auto",
) -> List[List[str]]:
    """
    Génère des templates de contexte dynamiquement comme dans MEMIT original.
    
    Args:
        model: Le modèle de langage
        tok: Tokenizer
        language: 'fr', 'en', ou 'auto' (détecte automatiquement)
    
    Returns:
        Liste de listes de templates [["{}"]] + [[generated templates]]
    """
    global CONTEXT_TEMPLATES_CACHE
    
    if CONTEXT_TEMPLATES_CACHE is not None:
        return CONTEXT_TEMPLATES_CACHE
    
    # Détection automatique de la langue basée sur le nom du modèle
    if language == "auto":
        model_name = getattr(model.config, '_name_or_path', '').lower()
        if any(x in model_name for x in ['fr', 'french', 'camembert', 'flaubert']):
            language = "fr"
        else:
            # Par défaut, on utilise les deux pour plus de diversité
            language = "both"
    
    # Prompts de départ selon la langue
    if language == "fr":
        starter_prompts = ["Le", "Donc", "Parce que", "Je", "Vous"]
    elif language == "en":
        starter_prompts = ["The", "Therefore", "Because", "I", "You"]
    else:  # both
        starter_prompts = ["Le", "Donc", "Parce que", "The", "Therefore"]
    
    print(f"Generating dynamic context templates ({language})...")
    
    # Générer des contextes
    generated = generate_fast(
        model,
        tok,
        starter_prompts,
        n_gen_per_prompt=1,  # 1 génération par prompt = 5 templates
        max_out_len=10,      # Contextes courts
        top_k=5,
    )
    
    # Nettoyer et formater les templates
    templates = []
    for text in generated:
        # Nettoyer le texte
        text = text.strip()
        # Remplacer les accolades qui pourraient interférer avec format()
        text = text.replace("{", " ").replace("}", " ")
        # Ajouter le placeholder pour le prompt
        template = f"{text}. {{}}"
        templates.append(template)
    
    # Structure: [[direct template]] + [[generated templates]]
    CONTEXT_TEMPLATES_CACHE = [["{}"]] + [templates]
    
    print(f"Generated {len(templates)} context templates:")
    for t in templates[:3]:
        print(f"  - {t[:50]}...")
    
    return CONTEXT_TEMPLATES_CACHE


def reset_context_templates():
    """Réinitialise le cache des templates."""
    global CONTEXT_TEMPLATES_CACHE
    CONTEXT_TEMPLATES_CACHE = None


def generate_fast(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
) -> List[str]:
    """
    Génération de texte parallélisée avec échantillonnage top-k.
    Utilise le device du modèle.
    
    Args:
        model: Modèle de langage
        tok: Tokenizer
        prompts: Liste de prompts de départ
        n_gen_per_prompt: Nombre de générations par prompt
        top_k: Paramètre de sampling top-k
        max_out_len: Longueur maximale de sortie
    
    Returns:
        Liste de textes générés
    """
    # Déterminer le device du modèle
    device = next(model.parameters()).device
    
    # Configurer le tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # Important pour la génération
    
    # Préparer les prompts
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(device)
    input_ids = inp_tok["input_ids"]
    attention_mask = inp_tok["attention_mask"]
    batch_size = input_ids.size(0)
    
    # Génération avec cache KV pour l'efficacité
    past_key_values = None
    cur_context = slice(0, attention_mask.sum(1).min().item())
    
    with torch.no_grad():
        while input_ids.size(1) < max_out_len:
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = model_out.logits
            past_key_values = model_out.past_key_values
            
            # Échantillonnage top-k
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1, keepdim=True)
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)
            
            # Étendre les tenseurs si nécessaire
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)],
                    dim=1,
                )
                input_ids = torch.cat(
                    [input_ids, input_ids.new_full((batch_size, 1), tok.pad_token_id)],
                    dim=1,
                )
            
            # Insérer les nouveaux tokens
            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue
                
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1
            
            cur_context = slice(cur_context.stop, cur_context.stop + 1)
    
    # Décoder les résultats
    txt = [tok.decode(x, skip_special_tokens=True) for x in input_ids]
    
    # Nettoyer
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("\n", " ")
        .strip()
        for x in txt
    ]
    
    return txt
