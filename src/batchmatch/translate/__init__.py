"""Translation search module

This module provides FFT-based translation registration methods for finding
the translation offset between a moving image and a reference image.

Examples:
    >>> from batchmatch.translate import build_translation_stage
    >>> search = build_translation_stage("ncc")
    >>> moving = search(reference, moving)
    >>> moving.translation_results.tx

Available methods:
    - cc: Cross-correlation.
    - gcc: Gradient cross-correlation.
    - ncc: Normalized cross-correlation.
    - pc: Phase correlation.
    - gpc: Gradient phase correlation. 
    - ngf: Normalized gradient fields. (p=2, counts contributions of gradients aligned and anti-aligned with reference)
    - gngf: Generalized normalized gradient fields (for powers other than 2).
"""
from __future__ import annotations

from .stage import (
    TranslationSearchStage,
    translation_registry,
    # Buffer specification classes
    BufferShapeType,
    MovingBufferSpec,
    # Individual stage classes
    CrossCorrelationStage,
    GradientCrossCorrelationStage,
    MaskedNCCStage,
    PhaseCorrelationStage,
    GradientPhaseCorrelationStage,
    NormalizedGradientFieldsStage,
    GeneralizedNGFStage,
)

# Pipeline specification and builder
from .pipeline import (
    TranslationSearchSpec,
    build_translation_stage,
    coerce_translation_search_spec,
    get_translation_stage_class,
    list_translation_methods,
)
from .config import (
    TranslationConfig,
    CCTranslationConfig,
    MeanCCTranslationConfig,
    GCCTranslationConfig,
    NCCTranslationConfig,
    PCTranslationConfig,
    GPCTranslationConfig,
    NGFTranslationConfig,
    GNGFTranslationConfig,
)

__all__ = [
    # Stage base and registry
    "TranslationSearchStage",
    "translation_registry",
    # Buffer specification classes
    "BufferShapeType",
    "MovingBufferSpec",
    # Stage classes
    "CrossCorrelationStage",
    "GradientCrossCorrelationStage",
    "MaskedNCCStage",
    "PhaseCorrelationStage",
    "GradientPhaseCorrelationStage",
    "NormalizedGradientFieldsStage",
    "GeneralizedNGFStage",
    # Pipeline spec and builder
    "TranslationSearchSpec",
    "build_translation_stage",
    "coerce_translation_search_spec",
    "get_translation_stage_class",
    "list_translation_methods",
    # Typed config classes
    "TranslationConfig",
    "CCTranslationConfig",
    "MeanCCTranslationConfig",
    "GCCTranslationConfig",
    "NCCTranslationConfig",
    "PCTranslationConfig",
    "GPCTranslationConfig",
    "NGFTranslationConfig",
    "GNGFTranslationConfig",
]
