"""Evaluate generated crystal structures produced by MatterGen.

This script loads the structures from a generation results directory
(`generated_crystals.extxyz` or a zip) and computes evaluation metrics
using MatterGen's evaluation framework. It optionally performs a fast
relaxation (if a potential is available) before computing metrics.

Outputs:
 - metrics.json: aggregated metric values
 - relaxed_structures.extxyz (optional if relaxation enabled)

Minimal usage (structure-only metrics, no relaxation):
    python -m generation.evaluate --results-dir path/to/results

With relaxation using default ML potential path:
    python -m generation.evaluate --results-dir path/to/results --relax --potential ./potential.pt

NOTE: For large batches, relaxation can be costly; disable with `--no-relax`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from pymatgen.core.structure import Structure
    import ase.io
    from pymatgen.io.ase import AseAtomsAdaptor
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependencies. Install with: pip install pymatgen ase") from e

# Fallback: add mattergen source path if not installed as a package
import sys
_ROOT = Path(__file__).resolve().parents[2]
_MG_SRC = _ROOT / "mattergen"
if _MG_SRC.exists() and str(_MG_SRC) not in sys.path:
    sys.path.append(str(_MG_SRC))
try:
    from mattergen.evaluation.evaluate import evaluate as mg_evaluate
except ModuleNotFoundError as e:
    if e.name == "mattergen":
        raise SystemExit("MatterGen package not found. Ensure source at 'mattergen/' or install it.") from e
    else:
        # Surface the missing dependency name for easier debugging
        raise SystemExit(f"Missing dependency '{e.name}' required by MatterGen. Please install it.") from e

DEFAULT_RESULTS_FILE = "generated_crystals.extxyz"


def _load_structures(results_dir: Path) -> list[Structure]:
    extxyz = results_dir / DEFAULT_RESULTS_FILE
    if not extxyz.exists():
        raise SystemExit(f"Results file not found: {extxyz}")
    atoms_list = ase.io.read(extxyz, index=":", format="extxyz")
    return [AseAtomsAdaptor.get_structure(a) for a in atoms_list]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate generated crystal structures")
    p.add_argument("--results-dir", type=lambda v: Path(v).expanduser().resolve(), required=True, help="Directory containing generated_crystals.extxyz")
    p.add_argument("--output-dir", type=lambda v: Path(v).expanduser().resolve(), default=None, help="Directory to write metrics (default: results-dir)")
    p.add_argument("--relax", action="store_true", help="Relax structures before metric computation")
    p.add_argument("--no-relax", action="store_true", help="Disable relaxation (overrides --relax)")
    p.add_argument("--potential", type=str, default=None, help="Path to ML potential for relaxation (optional)")
    p.add_argument("--save-structures", action="store_true", help="Save relaxed structures if relaxation performed")
    return p


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)
    if ns.no_relax:
        ns.relax = False
    return ns


def run(options: argparse.Namespace) -> Path:
    results_dir: Path = options.results_dir
    output_dir = options.output_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    structures = _load_structures(results_dir)
    print(f"[evaluate-gen] Loaded {len(structures)} structures from {results_dir}")
    relaxed_output = None
    if options.relax and options.save_structures:
        relaxed_output = str(output_dir / "relaxed_structures.extxyz")

    # NOTE:
    # We intentionally do NOT pass an explicit `reference` here.
    # MatterGen's `evaluate` function will then use its default
    # reference dataset (currently based on reference_MP2020correction),
    # which produces metrics with fields like:
    #   avg_energy_above_hull_per_atom, avg_rmsd_from_relaxation,
    #   frac_novel_unique_stable_structures, frac_stable_structures,
    #   frac_successful_jobs, avg_comp_validity, avg_structure_comp_validity,
    #   avg_structure_validity, frac_novel_structures, frac_novel_systems,
    #   frac_novel_unique_structures, frac_unique_structures,
    #   frac_unique_systems, precision, recall, ...
    # i.e. the same schema as the example JSON the user provided.
    metrics = mg_evaluate(
        structures=structures,
        relax=options.relax,
        potential_load_path=options.potential,
        save_as=str(metrics_path),
        structures_output_path=relaxed_output,
    )
    # Store a lightweight summary
    summary = {
        "count": len(structures),
        "metrics": metrics,
        "relaxed": options.relax,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[evaluate-gen] Metrics written to {metrics_path}")
    return metrics_path


def main(argv: Iterable[str] | None = None) -> None:
    opts = parse_args(argv)
    run(opts)


if __name__ == "__main__":  # pragma: no cover
    import sys
    main(sys.argv[1:])
