"""
Utilitaires pour instrumenter un modèle PyTorch.
Permet d'intercepter les activations et de les modifier.

Trace: hook une couche à la fois
TraceDict: hook plusieurs couches simultanément
"""

import contextlib
import copy
import inspect
from collections import OrderedDict
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn


class Trace(contextlib.AbstractContextManager):
    """
    Context manager pour capturer les entrées/sorties d'une couche.
    
    Exemple:
        with Trace(model, 'model.layers.5.mlp') as tr:
            _ = model(inputs)
            hidden = tr.output  # Activations de la couche
    """

    def __init__(
        self,
        module: nn.Module,
        layer: Optional[str] = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Optional[Callable] = None,
        stop: bool = False,
    ):
        self.layer = layer
        self.stop = stop
        
        if layer is not None:
            module = get_module(module, layer)
        
        retainer = self

        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and type is not None and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    Context manager pour capturer les entrées/sorties de plusieurs couches.
    
    Exemple:
        with TraceDict(model, ['layer1', 'layer2']) as tr:
            _ = model(inputs)
            out1 = tr['layer1'].output
    """

    def __init__(
        self,
        module: nn.Module,
        layers: List[str],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Optional[Callable] = None,
        stop: bool = False,
    ):
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and type is not None and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


class StopForward(Exception):
    """Exception pour arrêter la propagation forward après une couche."""
    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """Copie récursive de tenseurs avec options de détachement/clonage."""
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, clone, detach, retain_grad) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v, clone, detach, retain_grad) for v in x])
    else:
        return x  # Types non-tensor retournés tels quels


def get_module(model: nn.Module, name: str) -> nn.Module:
    """Récupère un sous-module par son nom (e.g., 'model.layers.5.mlp')."""
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(f"Module '{name}' not found in model")


def get_parameter(model: nn.Module, name: str) -> torch.nn.Parameter:
    """Récupère un paramètre par son nom."""
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(f"Parameter '{name}' not found in model")


def set_requires_grad(requires_grad: bool, *models):
    """Active/désactive requires_grad pour tous les paramètres des modèles."""
    for model in models:
        if isinstance(model, nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad


def invoke_with_optional_args(fn, *args, **kwargs):
    """Appelle une fonction avec seulement les arguments qu'elle accepte."""
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    used_pos = 0
    
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            # Use default if available
            defaulted_pos = len(argspec.args) - (
                0 if not argspec.defaults else len(argspec.defaults)
            )
            if i >= defaulted_pos:
                pass_args.append(argspec.defaults[i - defaulted_pos])
    
    # Pass remaining kwargs if function accepts them
    pass_kw = {
        k: v for k, v in kwargs.items()
        if k not in used_kw and (k in (argspec.kwonlyargs or []) or argspec.varkw is not None)
    }
    
    return fn(*pass_args, **pass_kw)


