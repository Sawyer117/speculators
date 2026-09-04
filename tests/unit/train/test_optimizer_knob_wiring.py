"""Every optimizer knob must exist on all THREE config surfaces, not just one.

WHY. `muon_hybrid_ns` was added to `train/config/schema.py` alone. That surface is a
separate entry point the NPU launcher never touches, so two things broke in sequence:

  1. the launcher emitted `--muon-hybrid-ns` and `scripts/train.py` -- which has its own
     hand-written argparse -- answered `unrecognized arguments`, killing all 8 ranks;
  2. after the flag was accepted, `build_optimizers` read `config.muon_hybrid_ns` off
     `TrainerConfig`, a NamedTuple that still had no such field, so EVERY muon run died
     with AttributeError whether or not the knob was passed.

Both failures are static and cost a multi-hour training slot each. These tests read the
sources rather than importing the trainer, so they run without torch_npu present.
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TRAIN_PY = REPO / "scripts" / "train.py"
TRAINER_PY = REPO / "src" / "speculators" / "train" / "trainer.py"
OPTIMIZERS_PY = REPO / "src" / "speculators" / "train" / "optimizers.py"
LAUNCHER = REPO / "examples" / "ascend_npu_dflash" / "train_dsv4_dspark.sh"


def _trainer_config_fields() -> set[str]:
    """Field names on the ``TrainerConfig`` NamedTuple, read from the AST."""
    tree = ast.parse(TRAINER_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrainerConfig":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    pytest.fail("TrainerConfig not found in trainer.py")
    return set()  # unreachable, keeps the type checker happy


def _train_py_flags() -> set[str]:
    """Long flags registered on scripts/train.py's own parser."""
    return set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', TRAIN_PY.read_text()))


def _config_attrs_read_by(path: Path) -> set[str]:
    """Attribute names read off a name called ``config`` in ``path``."""
    tree = ast.parse(path.read_text())
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    }


def test_build_optimizers_only_reads_fields_trainer_config_has():
    """The AttributeError that killed every muon run, caught statically.

    `build_optimizers` is handed a `TrainerConfig`, so every `config.X` it reads must be
    a field on that NamedTuple.
    """
    missing = _config_attrs_read_by(OPTIMIZERS_PY) - _trainer_config_fields()
    assert not missing, (
        f"build_optimizers reads {sorted(missing)} off config, but TrainerConfig has no "
        "such field -- every run using that optimizer will die with AttributeError. Add "
        "the field to the TrainerConfig NamedTuple in train/trainer.py."
    )


@pytest.mark.parametrize(
    "field",
    sorted(f for f in _trainer_config_fields() if f.startswith("muon_")),
)
def test_every_muon_field_has_a_train_py_flag(field: str):
    """A knob nobody can set is dead weight; worse, the launcher may emit its flag."""
    flag = "--" + field.replace("_", "-")
    assert flag in _train_py_flags(), (
        f"TrainerConfig.{field} has no {flag} in scripts/train.py's parser. That parser "
        "is hand-written and does NOT come from train/config/schema.py, so adding the "
        "field to the schema alone is not enough."
    )


def _launcher_emitted_flags() -> set[str]:
    """Flags the launcher actually hands to train.py.

    Only two places matter, and scanning the whole file instead sweeps up pip flags and
    shell fragments from the comments: the ``EXTRA="$EXTRA --flag"`` accumulator, and the
    backslash-continued argument lines of the torchrun command itself. Flags built by
    string interpolation (``--init-$X``) are skipped -- they cannot be checked statically.
    """
    flags: set[str] = set()
    for raw in LAUNCHER.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if "EXTRA=" not in line and not line.startswith("--"):
            continue
        for flag in re.findall(r"(--[a-z0-9][a-z0-9-]*)", line):
            if not flag.endswith("-"):  # drop interpolated stems
                flags.add(flag)
    return flags


def test_launcher_only_emits_flags_train_py_accepts():
    """The launcher shells out to train.py; an unknown flag kills all ranks at startup."""
    known = _train_py_flags()
    unknown = {
        f
        for f in _launcher_emitted_flags()
        # `--no-x` is argparse's BooleanOptionalAction partner for a registered `--x`.
        if f not in known and "--" + f.removeprefix("--no-") not in known
    }
    assert not unknown, (
        f"train_dsv4_dspark.sh emits {sorted(unknown)}, which scripts/train.py's parser "
        "does not define -- torchrun fails with 'unrecognized arguments' before the "
        "model is even built."
    )
