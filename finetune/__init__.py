"""Finetune module utilities."""

from __future__ import annotations

from .local import FinetuneOptions, run_finetune

__all__ = ["FinetuneOptions", "run_finetune"]


def __getattr__(name: str):  # pragma: no cover - helper for optional imports
    if name in {"FinetuneSubmitOptions", "submit_finetune_job"}:
        from .remote import FinetuneSubmitOptions, submit_finetune_job

        return {  # type: ignore[return-value]
            "FinetuneSubmitOptions": FinetuneSubmitOptions,
            "submit_finetune_job": submit_finetune_job,
        }[name]
    raise AttributeError(name)
