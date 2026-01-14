"""
Hyperparamètres pour MEMIT.
"""

from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class MEMITHyperParams:
    """Configuration pour l'algorithme MEMIT."""
    
    # Identité
    alg_name: str = "MEMIT"
    model_name: str = ""
    
    # Device
    device: int = 0
    
    # Couches ciblées (MEMIT utilise PLUSIEURS couches)
    layers: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14])
    v_loss_layer: int = 25
    
    # Optimisation du vecteur z
    v_num_grad_steps: int = 20
    v_lr: float = 0.5
    v_weight_decay: float = 1e-3
    clamp_norm_factor: float = 4.0
    kl_factor: float = 0.0625
    
    # Covariance
    mom2_adjustment: bool = True
    mom2_update_weight: float = 20000.0  # Poids pour la régularisation
    mom2_dataset: str = "wikipedia_fr"
    mom2_n_samples: int = 1000
    mom2_dtype: str = "float32"
    
    # Stratégie de token
    fact_token: str = "subject_last"
    
    # Templates de modules (Ministral 3B multimodal)
    rewrite_module_tmp: str = "model.language_model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.language_model.layers.{}"
    mlp_module_tmp: str = "model.language_model.layers.{}.mlp"
    attn_module_tmp: str = "model.language_model.layers.{}.self_attn"
    ln_f_module: str = "model.language_model.norm"
    lm_head_module: str = "model.language_model.lm_head"
    embed_module: str = "model.language_model.embed_tokens"  # For causal tracing
    
    # Batch size pour le calcul des vecteurs k
    batch_size: int = 32
    
    # Stats directory
    stats_dir: str = "./data/stats"
    
    @classmethod
    def from_yaml(cls, path: str) -> "MEMITHyperParams":
        """Charge les hyperparamètres depuis un fichier YAML."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})

