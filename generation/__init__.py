"""Generation module: local run + remote submission."""

from .local import GenerationOptions, run_generation
from .remote import submit_generation_job

__all__ = ["GenerationOptions", "run_generation", "submit_generation_job"]
