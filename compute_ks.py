"""
Calcul des vecteurs clés (k) pour chaque couche MEMIT.
"""

from typing import Dict, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from . import nethook, repr_tools
from .memit_hparams import MEMITHyperParams


def compute_ks(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[List[str]],
) -> torch.Tensor:
    """Calcule les vecteurs clés k pour une couche donnée."""    

    all_contexts = []
    all_words = []
    
    for request in requests:
        for context_types in context_templates:
            for context in context_types:
                all_contexts.append(context.format(request["prompt"]))
                all_words.append(request["subject"])
    

    tok.padding_side = "right"
    layer_ks = get_module_input_at_words(
        model,
        tok,
        layer,
        context_templates=all_contexts,
        words=all_words,
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
        batch_size=hparams.batch_size,
    )
    

    num_contexts = sum(len(ct) for ct in context_templates)
    ans = []
    
    for i in range(0, layer_ks.size(0), num_contexts):

        req_ks = layer_ks[i:i + num_contexts]
        ans.append(req_ks.mean(0))
    
    return torch.stack(ans, dim=0)


def get_module_input_at_words(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    batch_size: int = 32,
) -> torch.Tensor:
    device = next(model.parameters()).device
    

    idxs = []
    for i, (context, word) in enumerate(zip(context_templates, words)):
        idx = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[context],
            words=[word],
            subtoken=fact_token_strategy[len("subject_"):] if "subject_" in fact_token_strategy else "last",
        )[0][0]
        idxs.append([idx])
    

    contexts = [context.format(word) for context, word in zip(context_templates, words)]
    
    module_name = module_template.format(layer)
    results = []

    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i:i + batch_size]
        batch_idxs = idxs[i:i + batch_size]
        
        inputs = tok(batch_contexts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=True,
                retain_output=False,
            ) as tr:
                model(**inputs)
            

            inp = tr.input
            if isinstance(inp, tuple):
                inp = inp[0]
            
            for j, idx_list in enumerate(batch_idxs):
                seq_len = inp.shape[1]
                safe_idx = idx_list[0]
                if safe_idx < 0:
                    safe_idx = seq_len + safe_idx
                safe_idx = min(safe_idx, seq_len - 1)
                safe_idx = max(safe_idx, 0)
                
                results.append(inp[j, safe_idx].float())
    
    return torch.stack(results, dim=0)

