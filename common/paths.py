"""Locations of downloaded data and checkpoints.

Configs and scripts refer to files with repo-relative paths such as
``data/sparse_datasets/hopper_medium_expert_sparse_78.pkl`` or
``checkpoints/hopper_medium_expert_sparse/realnvp_42/realnvp_42``. ``resolve``
turns those into absolute paths, so code works no matter which folder it is
launched from (``mbpo_based/``, ``mobile_based/`` or the repo root).

Download the files first with ``python scripts/download_assets.py``.

Environment variables (optional):
    GORMPO_DATA_DIR  where ``data/...`` paths point (default: <repo>/data)
    GORMPO_CKPT_DIR  where ``checkpoints/...`` paths point (default: <repo>/checkpoints)
"""
import glob
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("GORMPO_DATA_DIR", os.path.join(REPO_ROOT, "data"))
CKPT_DIR = os.environ.get("GORMPO_CKPT_DIR", os.path.join(REPO_ROOT, "checkpoints"))

# Hugging Face repos holding the paper's data and checkpoints.
HF_DATA_REPO = "aysin/sparse_medium_expert_D4RL"
HF_MODEL_REPO = "aysin/GORMPO"


def resolve(path):
    """Return an absolute path for ``path``.

    Absolute paths are returned unchanged. ``data/...`` and ``checkpoints/...``
    are placed under DATA_DIR and CKPT_DIR; any other relative path is taken
    relative to the repo root. Empty values and the special values ``true`` and
    ``auto`` are returned as they are.
    """
    if path is None:
        return None
    path = str(path)
    if path == "" or path.lower() in ("true", "auto", "none"):
        return path
    path = os.path.expanduser(os.path.expandvars(path))
    if os.path.isabs(path):
        return path
    norm = path.replace("\\", "/")
    if norm.startswith("./"):
        norm = norm[2:]
    for prefix, root in (("data/", DATA_DIR), ("checkpoints/", CKPT_DIR)):
        if norm.startswith(prefix):
            return os.path.join(root, norm[len(prefix):])
    return os.path.join(REPO_ROOT, norm)


def env_name(task):
    """'hopper-medium-expert-v2' -> 'hopper'."""
    return task.lower().split("-")[0]


def paper_run_dir(task, base, density_model, seed):
    """Folder of a downloaded paper run.

    Args:
        task: D4RL task, e.g. 'hopper-medium-expert-v2'.
        base: 'mbpo_based' or 'mobile_based'.
        density_model: 'kde', 'vae', 'realnvp', 'diffusion' or 'neuralODE'.
        seed: 42, 123 or 456.

    Looks for checkpoints/<env>_medium_expert_sparse/<base>/<density_model>/seed_<seed>_*
    (MBPO) or seed_<seed>&* (MOBILE) and raises FileNotFoundError if there is not
    exactly one match.
    """
    root = os.path.join(CKPT_DIR, f"{env_name(task)}_medium_expert_sparse", base, density_model)
    matches = sorted(glob.glob(os.path.join(root, f"seed_{seed}_*")) + glob.glob(os.path.join(root, f"seed_{seed}&*")))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one paper run for task={task} base={base} density_model={density_model} seed={seed} "
            f"under {root}, found {len(matches)}. Run `python scripts/download_assets.py` or pass an explicit path."
        )
    return matches[0]


# Argument names that hold file or folder paths in the training / OOD-test scripts.
PATH_ARG_NAMES = (
    "data_path", "dataset_path", "model_save_path", "save_path", "model_path", "model_dir",
    "base_path", "out", "npz", "test_npz", "train_npz", "classifier_model_name", "output_dir",
    "ood_test_dir", "scheduler_path", "scheduler_dir",
)


def resolve_args(args, names=PATH_ARG_NAMES):
    """Resolve every path-like attribute of an argparse namespace in place."""
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if isinstance(value, str):
                setattr(args, name, resolve(value))
    return args
