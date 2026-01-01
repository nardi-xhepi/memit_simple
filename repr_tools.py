"""
Utilitaires pour extraire les représentations de tokens.
Utilisé pour calculer les vecteurs u et v dans ROME.
"""

from copy import deepcopy
from typing import List, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from . import nethook


def get_reprs_at_word_tokens(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Récupère les représentations du dernier token d'un mot dans un contexte.
    
    Args:
        model: Le modèle
        tok: Le tokenizer
        context_templates: Templates de contexte avec {} pour le mot
        words: Mots à substituer
        layer: Numéro de la couche
        module_template: Template du module (e.g., 'model.layers.{}.mlp')
        subtoken: 'first', 'last' ou 'first_after_last'
        track: 'in', 'out' ou 'both'
    
    Returns:
        Tenseur des représentations moyennées
    """
    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    tok: PreTrainedTokenizer,
    context_templates: List[str],
    words: List[str],
    subtoken: str,
) -> List[List[int]]:
    """
    Calcule les indices des tokens d'un mot dans des templates.
    
    IMPLÉMENTATION ORIGINALE DE MEMIT (github.com/kmeng01/memit):
    Tokenise séparément le préfixe, le mot, et le suffixe pour calculer
    les indices de manière robuste.
    
    Args:
        tok: Tokenizer
        context_templates: Templates avec {} pour insertion
        words: Mots à insérer
        subtoken: 'first', 'last' ou 'first_after_last'
    
    Returns:
        Liste d'indices pour chaque template
    """
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "Un seul {} par template supporté"

    # Découper les templates en préfixe/suffixe
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes = [tmp[:fill_idxs[i]] for i, tmp in enumerate(context_templates)]
    suffixes = [tmp[fill_idxs[i] + 2:] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pré-traiter les tokens (gérer les espaces)
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " ", f"Prefix should end with space: '{prefix}'"
            prefix = prefix[:-1]
            
            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokeniser pour déterminer les longueurs
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    
    # Tokeniser tous les éléments ensemble (batched)
    batch_tok = tok([*prefixes, *words, *suffixes], add_special_tokens=False)["input_ids"]
    prefixes_tok = batch_tok[0:n]
    words_tok = batch_tok[n:2*n]
    suffixes_tok = batch_tok[2*n:3*n]
    
    # Calculer les longueurs (ce sont directement des listes d'IDs)
    prefixes_len = [len(el) for el in prefixes_tok]
    words_len = [len(el) for el in words_tok]
    suffixes_len = [len(el) for el in suffixes_tok]

    # Calculer les indices des tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # Si le suffixe est vide, pas de "premier token après le dernier"
            # Donc on retourne juste le dernier token du mot
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Subtoken inconnu: {subtoken}")




def get_reprs_at_idxs(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Exécute le modèle et récupère les représentations aux indices spécifiés.
    
    Args:
        model: Le modèle
        tok: Tokenizer
        contexts: Textes de contexte
        idxs: Indices des tokens à extraire
        layer: Numéro de couche
        module_template: Template du module
        track: 'in', 'out' ou 'both'
    
    Returns:
        Tenseurs des représentations moyennées
    """
    assert track in {"in", "out", "both"}
    both = track == "both"
    tin = track == "in" or both
    tout = track == "out" or both
    
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}
    device = next(model.parameters()).device

    def _process(cur_repr, batch_idxs, key):
        cur_repr = cur_repr[0] if isinstance(cur_repr, tuple) else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            # Gérer les indices négatifs et hors limites
            seq_len = cur_repr.shape[1]
            safe_idxs = []
            for idx in idx_list:
                if idx < 0:
                    idx = seq_len + idx
                idx = min(idx, seq_len - 1)
                idx = max(idx, 0)
                safe_idxs.append(idx)
            to_return[key].append(cur_repr[i][safe_idxs].mean(0))

    # Traitement par batch
    batch_size = 512
    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i:i + batch_size]
        batch_idxs = idxs[i:i + batch_size]
        
        # Tokeniser avec padding à droite pour garder les indices cohérents
        tok.padding_side = "right"
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
