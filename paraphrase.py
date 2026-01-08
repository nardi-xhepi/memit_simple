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
    
    # Instruction for paraphrasing
    instruction = f"""Reformule cette phrase de {n_paraphrases} façons différentes en gardant exactement le même sens. Garde le mot "{subject}" dans chaque reformulation.

Phrase originale: "{original_sentence}"

Reformulations:
1."""

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    inputs = tok(instruction, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    
    generated = tok.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the numbered list
    paraphrases = parse_paraphrases(generated, subject, n_paraphrases)
    
    # Always include original prompt
    result = [prompt] + paraphrases
    
    # Remove duplicates while preserving order
    seen = set()
    unique_result = []
    for p in result:
        if p not in seen:
            seen.add(p)
            unique_result.append(p)
    
    print(f"  Generated {len(unique_result)} prompt variations for '{subject}'")
    for i, p in enumerate(unique_result):
        print(f"    [{i}] {p.format(subject)}")
    
    PARAPHRASE_CACHE[cache_key] = unique_result
    return unique_result


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
            
            if subject.lower() in sentence.lower():
                # Replace subject with placeholder (case-insensitive)
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


def reset_paraphrase_cache():
    """Clear the paraphrase cache."""
    global PARAPHRASE_CACHE
    PARAPHRASE_CACHE = {}
