#!/usr/bin/env python3
"""End-to-end script: finetune -> evaluate -> generate.

Steps:
1. Finetune a base MatterGen model on provided train/val caches.
2. Evaluate baseline vs finetuned checkpoint metrics.
3. Generate structures using the finetuned checkpoint (fallback to baseline if generation fails).

The script reuses existing crygen_demo tooling and MatterGen APIs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

# Local imports (available because script placed in crygen_demo tree)
from finetune.local import FinetuneOptions, parse_args as parse_finetune_cli, run_finetune, build_overrides, compose_config
from finetune.evaluate_finetune import _format_metrics as format_metrics, _prepare_options as prepare_eval_options
from finetune.evaluate_finetune import build_overrides as eval_build_overrides, compose_config as eval_compose_config, init_adapter_lightningmodule_from_pretrained  # type: ignore

from mattergen.diffusion.run import maybe_instantiate
from omegaconf import open_dict
from pytorch_lightning.callbacks import LearningRateMonitor

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PIPELINE_ROOT = SCRIPT_DIR / "pipeline_run"
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "data"
DEFAULT_BASE_MODEL_DIR = SCRIPT_DIR.parent.parent / "mattergen" / "checkpoints" / "mattergen_base"
DEFAULT_NUM_GEN = 8


def _abs(path: Path) -> Path:
    return path.expanduser().resolve()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finetune + evaluate + generate pipeline")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Train/val cache root (contains train/ val/ folders)")
    p.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR, help="Baseline MatterGen model directory")
    p.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT, help="Where to write all artifacts")
    p.add_argument("--max-epochs", type=int, default=1, help="Finetune epochs (default: 1 for quick demo)")
    p.add_argument("--train-batch-size", type=int, default=32, help="Training batch size")
    p.add_argument("--val-batch-size", type=int, default=32, help="Validation batch size")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--devices", type=int, default=1, help="Accelerator devices (default: 1)")
    p.add_argument("--accelerator", choices=["auto", "gpu", "cpu"], default="auto", help="Lightning accelerator")
    p.add_argument("--precision", choices=["32", "16"], default="32", help="Training / eval precision")
    p.add_argument("--num-gen", type=int, default=DEFAULT_NUM_GEN, help="Number of structures to generate with finetuned model")
    p.add_argument("--use-wandb", action="store_true", help="Enable W&B logging if available")
    return p


def _finetune(options: argparse.Namespace) -> Path:
    pipeline_root = _abs(options.pipeline_root)
    ft_out = pipeline_root / "finetune_outputs"
    ckpt_path = ft_out / "checkpoints" / "last.ckpt"
    ft_opts = FinetuneOptions(
        data_root=_abs(options.data_root),
        model_dir=_abs(options.base_model_dir),
        output_dir=ft_out,
        max_epochs=options.max_epochs,
        train_batch_size=options.train_batch_size,
        val_batch_size=options.val_batch_size,
        num_workers=options.num_workers,
        devices=options.devices,
        accelerator=options.accelerator,
        strategy="auto",
        precision=options.precision,
        resume=None,
        use_wandb=options.use_wandb,
    )
    print(f"[finetune] output_dir={ft_out}")
    run_finetune(ft_opts)
    if not ckpt_path.exists():
        raise SystemExit(f"Finetuned checkpoint not found: {ckpt_path}")
    return ckpt_path


def _evaluate(options: argparse.Namespace, ckpt: Path) -> Dict[str, Any]:
    eval_root = _abs(options.pipeline_root) / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)
    eval_opts = prepare_eval_options(argparse.Namespace(
        data_root=_abs(options.data_root),
        model_dir=_abs(options.base_model_dir),
        batch_size=options.val_batch_size,
        num_workers=options.num_workers,
        devices=options.devices,
        accelerator=options.accelerator,
        precision=options.precision,
    ))
    overrides = build_overrides(eval_opts)
    cfg = compose_config(overrides)
    trainer = maybe_instantiate(cfg.trainer)
    datamodule = maybe_instantiate(cfg.data_module)
    if not trainer.logger:
        trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, LearningRateMonitor)]
    pl_module, lightning_cfg = init_adapter_lightningmodule_from_pretrained(cfg.adapter, cfg.lightning_module)
    from omegaconf import OmegaConf
    with open_dict(cfg):
        cfg.lightning_module = lightning_cfg
    print("[evaluate] baseline")
    base_metrics = trainer.validate(model=pl_module, datamodule=datamodule)
    print("[evaluate] finetuned")
    finetuned_metrics = trainer.validate(model=pl_module, datamodule=datamodule, ckpt_path=str(ckpt))
    # Summaries
    summary = {
        "baseline": base_metrics[0] if base_metrics else {},
        "finetuned": finetuned_metrics[0] if finetuned_metrics else {},
        "delta": {k: finetuned_metrics[0][k] - base_metrics[0].get(k, 0.0) for k in finetuned_metrics[0]} if base_metrics and finetuned_metrics else {},
    }
    (eval_root / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[evaluate] metrics written to", eval_root / "metrics.json")
    return summary


def _generate(options: argparse.Namespace, finetuned_ckpt: Path) -> Path:
    gen_root = _abs(options.pipeline_root) / "generation"
    gen_root.mkdir(parents=True, exist_ok=True)
    # We adapt run_generation.py fallback to accept a checkpoint by copying it into a temp model dir.
    adapted_model_dir = gen_root / "model_for_generation"
    if adapted_model_dir.exists():
        shutil.rmtree(adapted_model_dir)
    adapted_model_dir.mkdir(parents=True)
    # Copy base model config assets to ensure Hydra can find 'config'
    base_model_dir = _abs(options.base_model_dir)
    if base_model_dir.exists():
        for entry in base_model_dir.iterdir():
            # Skip original checkpoint files; we'll override with finetuned one
            if entry.is_file() and entry.suffix == ".ckpt":
                continue
            target = adapted_model_dir / entry.name
            try:
                if entry.is_dir():
                    shutil.copytree(entry, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, target)
            except Exception as copy_e:
                print(f"[generate] skipping copy of {entry} due to error: {copy_e}")
    # Place finetuned checkpoint (named last.ckpt)
    shutil.copy2(finetuned_ckpt, adapted_model_dir / "last.ckpt")
    # Use mattergen-generate only if directory resembles expected; this is a simplification.
    # If fails, we skip generation gracefully.
    num_gen = options.num_gen
    batch_size = min(4, num_gen)
    num_batches = (num_gen + batch_size - 1) // batch_size
    results_dir = gen_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mattergen-generate",
        str(results_dir),
        f"--model_path={adapted_model_dir}",
        f"--batch_size={batch_size}",
        f"--num_batches={num_batches}",
    ]
    print(f"[generate] running CLI: {' '.join(cmd)}")
    cli_ok = False
    try:
        subprocess.run(cmd, check=True)
        cli_ok = True
    except FileNotFoundError as e:
        print(f"[generate] mattergen-generate not found ({e}); will try Python API fallback.")
    except subprocess.CalledProcessError as e:
        print(f"[generate] CLI returned non-zero exit ({e}); will try Python API fallback.")
    except Exception as e:
        print(f"[generate] unexpected CLI error ({e}); will try Python API fallback.")

    if not cli_ok:
        try:
            print("[generate] attempting Python API fallback via mattergen.scripts.generate.main")
            from mattergen.scripts import generate as mg_generate  # type: ignore
            # Call main directly with parameters; this returns list[Structure]
            mg_generate.main(
                output_path=str(results_dir),
                model_path=str(adapted_model_dir),
                batch_size=batch_size,
                num_batches=num_batches,
                checkpoint_epoch="last",
                record_trajectories=True,
                sampling_config_path=str(Path("mattergen") / "sampling_conf"),
            )
            print("[generate] Python API fallback succeeded.")
        except Exception as api_e:
            print(f"[generate] Python API fallback failed: {api_e}; no structures generated.")
            # Create a marker file to signal failure
            (results_dir / "GENERATION_FAILED.txt").write_text(str(api_e))
    return results_dir


def _evaluate_generated_structures(gen_dir: Path) -> Dict[str, Any]:
    """Evaluate generated structures using MatterGen's evaluation utilities.

    This reuses the local `generation.evaluate` script logic but expects to run
    in an environment where MatterGen and its heavy dependencies are installed
    (e.g., the remote Bohrium container).
    """
    metrics_summary_path = gen_dir / "metrics_summary.json"
    metrics_path = gen_dir / "metrics.json"

    if metrics_summary_path.exists():
        try:
            return json.loads(metrics_summary_path.read_text())
        except Exception:
            pass

    # Try to import the local evaluation helper; if unavailable, skip gracefully.
    try:
        from generation import evaluate as gen_eval  # type: ignore
    except Exception as e:
        print(f"[gen-eval] generation.evaluate module not available ({e}); skipping generated-structure evaluation.")
        return {"error": "generation.evaluate module not available"}

    # Invoke its run() function programmatically.
    try:
        ns = argparse.Namespace(
            results_dir=gen_dir,
            output_dir=gen_dir,
            relax=False,
            no_relax=True,
            potential=None,
            save_structures=False,
        )
        gen_eval.run(ns)  # type: ignore[attr-defined]
    except SystemExit as e:
        print(f"[gen-eval] evaluation script exited: {e}; skipping.")
        return {"error": f"evaluation script exited: {e}"}
    except Exception as e:
        print(f"[gen-eval] evaluation failed: {e}; skipping.")
        return {"error": f"evaluation failed: {e}"}

    # Reload summary generated by generation.evaluate
    if metrics_summary_path.exists():
        try:
            return json.loads(metrics_summary_path.read_text())
        except Exception as e:
            print(f"[gen-eval] failed to read metrics_summary.json: {e}")

    # Fallback: if only metrics.json exists, wrap it.
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}
        return {"metrics_only": True, "metrics": metrics}

    return {"error": "no metrics file produced"}


def main(argv: Iterable[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    pipeline_root = _abs(args.pipeline_root)
    pipeline_root.mkdir(parents=True, exist_ok=True)

    ckpt = _finetune(args)
    metrics = _evaluate(args, ckpt)
    gen_dir = _generate(args, ckpt)
    gen_eval = _evaluate_generated_structures(gen_dir)

    summary = {
        "checkpoint": str(ckpt),
        "metrics": metrics,
        "generation_dir": str(gen_dir),
        "generation_eval": gen_eval,
    }
    (pipeline_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[pipeline] summary written to", pipeline_root / "summary.json")


if __name__ == "__main__":
    main(sys.argv[1:])
