"""Project-wide constants used by local/remote tooling."""

from __future__ import annotations

from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent.parent
CRYGEN_ROOT = TEST_ROOT.parent
MATTERGEN_ROOT = CRYGEN_ROOT / "mattergen"

DEFAULT_MODEL_CHECKPOINT = TEST_ROOT / "model" / "last.ckpt"

# Default locations for generation tasks
DEFAULT_GENERATION_RESULTS_DIR = TEST_ROOT / "results"
DEFAULT_GENERATION_MODEL_DIR = MATTERGEN_ROOT / "checkpoints" / "mattergen_base"

# Default locations for finetuning tasks
DEFAULT_FINETUNE_DATA_ROOT = CRYGEN_ROOT / "crygen_demo" / "data"
DEFAULT_FINETUNE_MODEL_DIR = MATTERGEN_ROOT / "checkpoints" / "mattergen_base"
DEFAULT_FINETUNE_OUTPUT_DIR = TEST_ROOT / "finetune" / "outputs"

# Remote execution defaults
DEFAULT_PROGRAM_ID = 29496
DEFAULT_REMOTE_IMAGE = "registry.dp.tech/dptech/dp/native/prod-26745/mattergen-custom:0.0.2"
DEFAULT_SCASS_TYPE = "1 * NVIDIA V100_32g"
DEFAULT_PLATFORM = "ali"
REMOTE_DEFAULT_MODEL_DIR = "./model"

# dpdispatcher always works relative to the submission folder
REMOTE_FORWARD_DIR = Path("./")
