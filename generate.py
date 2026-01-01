"""
Utilitaires pour générer du texte avec le modèle.
Adapté de l'implémentation officielle MEMIT.
"""

import unicodedata
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_fast(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 20,
) -> List[str]:
    """
    Génération auto-régressive rapide avec top-k sampling.
    
    Args:
        model: Le modèle de génération
        tok: Le tokenizer
        prompts: Liste de prompts à compléter
        n_gen_per_prompt: Nombre de générations par prompt
        top_k: Nombre de tokens top-k à considérer
        max_out_len: Longueur maximale de sortie
    
    Returns:
        Liste des textes générés
    """
    # Dérouler les prompts
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage avec KV cache
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # Ajouter le nouveau token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<s>", "")
        .replace("</s>", "")
        for x in txt
    ]

    return txt


def get_context_templates(model: PreTrainedModel, tok: PreTrainedTokenizer) -> List[List[str]]:
    """
    Génère dynamiquement des templates de contexte en utilisant le modèle.
    Adapté pour le français.
    
    Returns:
        Liste de listes de templates
    """
    import gc
    from transformers import Mistral3ForConditionalGeneration
    
    # Template de base
    templates = [["{}"]]
    
    # Le modèle principal peut être offloadé, on ne peut pas le déplacer
    # On va charger une COPIE sur CPU juste pour générer les templates
    print("  Chargement d'un modèle léger sur CPU pour génération de templates...")
    
    try:
        # Obtenir le nom du modèle
        model_name = model.config._name_or_path
        
        # Charger UNE SEULE FOIS sur CPU (le modèle restera petit en CPU-only)
        cpu_model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Plus léger
            device_map="cpu",  # Force CPU
            low_cpu_mem_usage=True,
        )
        
        # Prompts de départ en français
        french_prompts = ["Le", "Donc", "Parce que", "Je pense que", "Selon"]
        
        # Générer des continuations
        print("  Génération de templates de contexte dynamiques...")
        generated = generate_fast(
            cpu_model,
            tok,
            french_prompts,
            n_gen_per_prompt=1,
            max_out_len=10,
        )
        
        # Créer des templates en ajoutant "{}" à la fin
        new_templates = []
        for gen_text in generated:
            # Nettoyer et ajouter le placeholder
            clean_text = gen_text.strip().replace("{", " ").replace("}", " ")
            template = f"{clean_text}. {{}}"
            new_templates.append(template)
        
        templates.append(new_templates)
        
        print(f"  ✓ Templates générés: {templates}")
        
        # Libérer la mémoire du modèle CPU
        del cpu_model
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ⚠️  Erreur lors de la génération dynamique: {e}")
        print("  Utilisation de templates statiques comme fallback...")
        # Fallback sur templates statiques
        templates.append([
            "Selon les informations, {}",
            "Il est connu que {}",
            "D'après les sources, {}",
        ])
    
    return templates

