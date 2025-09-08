"""
Attribute generation module for compositionality analysis.

This module provides unified interfaces for extracting and encoding
attributes from different data modalities (sentences, words, knowledge graphs).
"""

from .base import BaseAttributeGenerator
from .sentence_attributes import SentenceConceptAttributes
from .word_attributes import WordMorphologyAttributes
from .kg_attributes import KGDemographicAttributes

__all__ = [
    'BaseAttributeGenerator',
    'SentenceConceptAttributes',
    'WordMorphologyAttributes',
    'KGDemographicAttributes'
]