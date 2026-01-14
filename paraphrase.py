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
    Generate paraphrases using grammatical patterns (hybrid approach).
    
    Uses common grammatical transformations that work for most fact types,
    since LLM-based paraphrasing is unreliable.
    """
    # Check cache
    cache_key = f"{prompt}|{subject}"
    if cache_key in PARAPHRASE_CACHE:
        return PARAPHRASE_CACHE[cache_key]
    
    result = [prompt]  # Always include original
    
    # Apply grammatical inversions based on common French patterns
    # These cover most fact-editing scenarios
    
    # Pattern: "La X de {} est" -> "{} a pour X", "{} a comme X"
    if " de {} " in prompt or " de {} " in prompt.lower():
        # Extract what comes before "de {}"
        # e.g., "La capitale de {} est" -> "capitale"
        inverted = create_subject_first_variant(prompt, subject)
        if inverted and inverted not in result:
            result.append(inverted)
    
    # Pattern: "Le/La X de {}" -> "{} a X"
    if prompt.startswith("Le ") or prompt.startswith("La "):
        inverted = create_possessive_variant(prompt, subject)
        if inverted and inverted not in result:
            result.append(inverted)
    
    # Add question form
    question = create_question_variant(prompt, subject)
    if question and question not in result:
        result.append(question)
    
    print(f"  Generated {len(result)} prompt variations for '{subject}'")
    for i, p in enumerate(result):
        print(f"    [{i}] {p.format(subject)}")
    
    PARAPHRASE_CACHE[cache_key] = result
    return result


def create_subject_first_variant(prompt: str, subject: str) -> Optional[str]:
    """Create a subject-first variant: 'La X de {} est' -> '{} a pour X'"""
    import re
    
    # Match patterns like "La capitale de {} est"
    match = re.match(r"^(Le |La |L')(\w+) de \{\}(.*)$", prompt, re.IGNORECASE)
    if match:
        article = match.group(1)
        noun = match.group(2)  # e.g., "capitale"
        rest = match.group(3)  # e.g., " est"
        
        # Create: "{} a pour capitale"
        return "{} a pour " + noun
    
    return None


def create_possessive_variant(prompt: str, subject: str) -> Optional[str]:
    """Create possessive variant: 'La couleur de {} est' -> '{} a la couleur'"""
    import re
    
    match = re.match(r"^(Le |La |L')(\w+) de \{\}(.*)$", prompt, re.IGNORECASE)
    if match:
        article = match.group(1).lower()
        noun = match.group(2)
        
        # Create: "{} a la couleur"
        return "{} a " + article + noun
    
    return None


def create_question_variant(prompt: str, subject: str) -> Optional[str]:
    """Create question variant: 'La capitale de {} est' -> 'Quelle est la capitale de {} ?'"""
    import re
    
    match = re.match(r"^(Le |La |L')(\w+) de \{\}(.*)$", prompt, re.IGNORECASE)
    if match:
        article = match.group(1)
        noun = match.group(2)
        
        # Determine question word based on article
        qword = "Quel" if article.lower() == "le " else "Quelle"
        return f"{qword} est {article.lower()}{noun} de {{}} ?"
    
    return None


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
