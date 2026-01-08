"""
MEMIT Simple - Mass-Editing Memory in a Transformer.
Version simplifiée adaptée pour Ministral 3B.

MEMIT étend ROME en éditant plusieurs couches simultanément,
ce qui améliore la généralisation aux reformulations.
"""

from .memit_main import apply_memit, execute_memit
from .memit_hparams import MEMITHyperParams
from .paraphrase import generate_paraphrases, reset_paraphrase_cache

__all__ = ['apply_memit', 'execute_memit', 'MEMITHyperParams', 'generate_paraphrases', 'reset_paraphrase_cache']


