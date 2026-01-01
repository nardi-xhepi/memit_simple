# MEMIT Simple

A simplified, standalone implementation of **MEMIT (Mass-Editing Memory in a Transformer)**, optimized for modern GPUs and tailored for Ministral 3B and similar models.

## Features

-   **Simplified Architecture**: Removed complex dependencies from the original ROME/MEMIT codebases.
-   **GPU Optimized**: Automatically handles device placement for heavy linear algebra operations.
-   **Ministral Support**: Specifically tested with Mistral/Ministral architectures.
-   **Easy Integration**: Installable as a standard Python package.

## Installation

```bash
git clone https://github.com/yourusername/memit_simple.git
cd memit_simple
pip install -e .
```

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from memit_simple import apply_memit, MEMITHyperParams

# 1. Load your model
model_name = "mistralai/Ministral-3B-Instruct-2512"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token

# 2. Define your edits
requests = [
    {
        "prompt": "{} est connu pour",
        "subject": "Picasso",
        "target_new": {"str": "être un célèbre musicien de jazz"},
    }
]

# 3. Load Hyperparameters
# You can create a JSON file or use a dictionary
hparams = MEMITHyperParams(
    alg_name="MEMIT",
    model_name=model_name,
    layers=[11, 12, 13, 14],  # Specific layers to edit
    v_loss_layer=25,          # Layer to calculate loss (usually usually last layer or close to it)
    mom2_update_weight=15000,
    # ... other params ...
)

# 4. Apply MEMIT
model_new, orig_weights = apply_memit(model, tok, requests, hparams)
```

## Citations

This code is based on the original work by Meng et al.:

```bibtex
@inproceedings{meng2022mass,
  title={Mass-Editing Memory in a Transformer},
  author={Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## License

MIT
