"""Ovi integration package — model contracts first, generation second.

Import order matters here: ``contracts`` must never depend on torch or on any
engine module, so the shape contracts stay verifiable on a CPU-only machine
with no weights mounted.
"""

from multigenai.ovi.contracts import (  # noqa: F401
    KNOWN_VARIANTS,
    ObservedShape,
    OviContractError,
    OviModelSpec,
    UnobservedContractError,
    conv_out_size,
    derive_seq_len,
    get_variant,
    load_dit_config,
    load_observed_shape_fixture,
    require_observed_shape,
)

__all__ = [
    "KNOWN_VARIANTS",
    "ObservedShape",
    "OviContractError",
    "OviModelSpec",
    "UnobservedContractError",
    "conv_out_size",
    "derive_seq_len",
    "get_variant",
    "load_dit_config",
    "load_observed_shape_fixture",
    "require_observed_shape",
]