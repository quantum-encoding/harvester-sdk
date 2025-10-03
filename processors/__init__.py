"""
Unified Parallel Processing - The Crown Jewel of the Harvester SDK

This module contains the legendary ParallelProcessor and its specialized descendants,
plus the complete pantheon of batch submitters for all major AI providers.

THE TRINITY IS COMPLETE: OpenAI, Anthropic, and Google Gemini.

Copyright (c) 2025 Quantum Encoding Ltd.
"""

from .parallel import (
    ParallelProcessor,
    ImageGenerationProcessor,
    TextGenerationProcessor,
    CodeRefactoringProcessor
)

# The Complete Batch Processing Pantheon
from .batch_submitter import (
    UnifiedBatchSubmitter,
    OpenAIBatchSubmitter,
    AnthropicBatchSubmitter
)

from .gemini_batch_submitter import GeminiBatchSubmitter

__all__ = [
    # Parallel Processing
    'ParallelProcessor',
    'ImageGenerationProcessor', 
    'TextGenerationProcessor',
    'CodeRefactoringProcessor',
    # Batch Processing Pantheon
    'UnifiedBatchSubmitter',
    'OpenAIBatchSubmitter',
    'AnthropicBatchSubmitter',
    'GeminiBatchSubmitter'
]

# Version of the Crown Jewel
__version__ = '3.0.0'  # The Trinity is complete