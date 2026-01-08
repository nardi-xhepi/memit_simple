"""
Paraphrase generation for MEMIT.
Uses the model itself to generate prompt variations for better generalization.
"""

import re
from typing import List, Tuple, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


# Cache for generated paraphrases
PARAPHRASE_CACHE = {}


def generate_paraphrases(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    prompt: str,
    subject: str,
    n_paraphrases: int = 3,
) -> List[str]:
    """
    Generate paraphrases of a prompt using the model itself.
    
    Args:
        model: The language model
        tok: Tokenizer
        prompt: Original prompt with {} placeholder for subject
        subject: The subject entity
        n_paraphrases: Number of paraphrases to generate
    
    Returns:
        List of paraphrased prompts (including original), each with {} placeholder
    """
    # Check cache
    cache_key = f"{prompt}|{subject}"
    if cache_key in PARAPHRASE_CACHE:
        return PARAPHRASE_CACHE[cache_key]
    
    device = next(model.parameters()).device
    original_sentence = prompt.format(subject)
    
    # Improved instruction - explicitly ask for grammatical inversions
    instruction = f"""Reformule cette phrase de {n_paraphrases} façons TRÈS DIFFÉRENTES grammaticalement.
RÈGLES:
- Garde EXACTEMENT le mot "[SUJET]" dans chaque reformulation
- IMPORTANT: Au moins une reformulation doit commencer par [SUJET]
- Exemples de structures variées:
  * "[SUJET] a pour X..." (sujet au début)
  * "X de [SUJET] est..." (sujet au milieu)  
  * "C'est [SUJET] qui..." (sujet au milieu)

Phrase: "{original_sentence.replace(subject, '[SUJET]')}"

Reformulations avec [SUJET]:
1. [SUJET]"""

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    inputs = tok(instruction, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    
    generated = tok.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the numbered list
    paraphrases = parse_paraphrases(generated, subject, n_paraphrases)
    
    # Always include original prompt first
    result = [prompt]
    
    # Add valid paraphrases
    for p in paraphrases:
        if p not in result and is_valid_template(p, subject):
            result.append(p)
    
    print(f"  Generated {len(result)} prompt variations for '{subject}'")
    for i, p in enumerate(result):
        print(f"    [{i}] {p.format(subject)}")
    
    PARAPHRASE_CACHE[cache_key] = result
    return result


def parse_paraphrases(generated_text: str, subject: str, expected_count: int) -> List[str]:
    """
    Parse numbered paraphrases from model output.
    
    Args:
        generated_text: Full model output
        subject: The subject to find and replace with {}
        expected_count: Number of paraphrases expected
    
    Returns:
        List of paraphrase templates with {} placeholder
    """
    paraphrases = []
    
    # Find lines starting with numbers
    lines = generated_text.split('\n')
    for line in lines:
        line = line.strip()
        # Match patterns like "1.", "1)", "1:", "- ", etc.
        match = re.match(r'^[\d]+[.):]\s*(.+)', line)
        if match:
            sentence = match.group(1).strip()
            # Remove quotes if present
            sentence = sentence.strip('"\'')
            
            # Replace [SUJET] marker with {}
            if '[SUJET]' in sentence:
                template = sentence.replace('[SUJET]', '{}')
                paraphrases.append(template)
            # Or try to find and replace the actual subject
            elif subject.lower() in sentence.lower():
                template = re.sub(
                    re.escape(subject), 
                    '{}', 
                    sentence, 
                    flags=re.IGNORECASE,
                    count=1
                )
                paraphrases.append(template)
                
            if len(paraphrases) >= expected_count:
                break
    
    return paraphrases


def is_valid_template(template: str, subject: str) -> bool:
    """Check if a template is valid for MEMIT."""
    # Must contain exactly one {} placeholder
    if template.count('{}') != 1:
        return False
    
    # Should not be too short
    if len(template) < 10:
        return False
    
    # The placeholder should be surrounded by spaces or at start/end
    # This ensures proper tokenization
    idx = template.find('{}')
    if idx > 0 and template[idx-1] not in ' \'"alan':
        return False
        
    return True


def reset_paraphrase_cache():
    """Clear the paraphrase cache."""
    global PARAPHRASE_CACHE
    PARAPHRASE_CACHE = {}
