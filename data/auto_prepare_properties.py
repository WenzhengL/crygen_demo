import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DATA_ROOT = Path(__file__).resolve().parent


def _detect_splits() -> List[str]:
    splits: List[str] = []
    for name in ("train", "val"):
        if (DATA_ROOT / name).is_dir():
            splits.append(name)
    return splits


def _iter_property_json() -> List[Tuple[str, Path]]:
    props: List[Tuple[str, Path]] = []
    for split in _detect_splits():
        for json_path in (DATA_ROOT / split).glob("*.json"):
            prop = json_path.stem
            props.append((prop, json_path))
    # keep stable order but unique by name
    seen = set()
    uniq: List[Tuple[str, Path]] = []
    for name, path in props:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, path))
    return uniq


def _load_values(json_path: Path) -> List:
    obj = json.loads(json_path.read_text())
    if "values" not in obj:
        raise KeyError(f"Missing 'values' in {json_path}")
    return obj["values"]


def _convert_values(values: List) -> np.ndarray:
    # Heuristic: if all items are "yes"/"no"-like, treat as binary label 0/1.
    lowered = [str(v).strip().lower() for v in values]
    uniq = {v for v in lowered}
    if uniq.issubset({"yes", "no", "true", "false", "0", "1"}):
        arr = np.array([1 if v in {"yes", "true", "1"} else 0 for v in lowered], dtype=np.int64)
        return arr
    # Otherwise, just try float conversion and fall back to raw
    try:
        arr = np.asarray(values, dtype=float)
        return arr
    except Exception:
        return np.asarray(values, dtype=object)


def prepare_properties() -> Dict[str, Dict[str, str]]:
    """Convert all split JSON property files to NPY and record dtypes.

    Returns a mapping: {property_name: {"dtype": str, "kind": "binary"|"numeric"|"other"}}.
    """

    meta: Dict[str, Dict[str, str]] = {}
    splits = _detect_splits()
    if not splits:
        raise FileNotFoundError(f"No train/ or val/ splits found under {DATA_ROOT}")

    for prop, _ in _iter_property_json():
        meta[prop] = {"dtype": "", "kind": ""}

    for split in splits:
        split_dir = DATA_ROOT / split
        for json_path in split_dir.glob("*.json"):
            prop = json_path.stem
            values = _load_values(json_path)
            arr = _convert_values(values)
            out_path = split_dir / f"{prop}.npy"
            np.save(out_path, arr)
            kind = "other"
            if arr.dtype == np.int64 and set(np.unique(arr).tolist()).issubset({0, 1}):
                kind = "binary"
            elif np.issubdtype(arr.dtype, np.floating):
                kind = "numeric"
            meta[prop] = {"dtype": str(arr.dtype), "kind": kind}
            print(f"[{split}] {prop}: wrote {out_path.name} with shape {arr.shape}, dtype={arr.dtype}, kind={kind}")

    meta_path = DATA_ROOT / "properties.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"Wrote property metadata to {meta_path}")
    return meta


if __name__ == "__main__":
    prepare_properties()
