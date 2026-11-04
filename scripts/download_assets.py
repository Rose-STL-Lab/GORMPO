"""Download the paper's datasets and checkpoints from Hugging Face.

    python scripts/download_assets.py                      # everything used in the paper
    python scripts/download_assets.py --envs hopper        # one environment
    python scripts/download_assets.py --what data density  # only datasets and density models
    python scripts/download_assets.py --models realnvp kde --what density mbpo

Files go to <repo>/data and <repo>/checkpoints (or $GORMPO_DATA_DIR / $GORMPO_CKPT_DIR)
with the same layout as the Hugging Face repos:

    data/sparse_datasets/<env>_medium_expert_sparse_<p>.pkl
    data/ood_test/<env>-medium-expert-v2/ood-distance-<d>-uniform.pkl
    checkpoints/<env>_medium_expert_sparse/<model>_<seed>/...           density models
    checkpoints/<env>_medium_expert_sparse/neuralODE/...                NeuralODE density model
    checkpoints/<env>_medium_expert_sparse/mbpo_based/<model>/seed_*/   MBPO-based GORMPO runs
    checkpoints/<env>_medium_expert_sparse/mobile_based/<model>/seed_*/ MOBILE-based GORMPO runs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import CKPT_DIR, DATA_DIR, HF_DATA_REPO, HF_MODEL_REPO  # noqa: E402

ENVS = ["halfcheetah", "hopper", "walker2d"]
MODELS = ["kde", "vae", "realnvp", "diffusion", "neuralODE"]
WHAT = ["data", "ood", "density", "mbpo", "mobile"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=ENVS, choices=ENVS)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="density models")
    p.add_argument("--what", nargs="+", default=WHAT, choices=WHAT,
                   help="data: sparse datasets, ood: OOD test sets, density: density models, "
                        "mbpo/mobile: trained GORMPO runs (policy + transition model)")
    p.add_argument("--intermediate", action="store_true",
                   help="also download intermediate training checkpoints (diffusion checkpoints/, NeuralODE checkpoint_epoch_*)")
    p.add_argument("--revision", default=None, help="git revision of both HF repos (default: main)")
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    if "data" in args.what or "ood" in args.what:
        pats = []
        if "data" in args.what:
            pats += [f"sparse_datasets/{e}_medium_expert_sparse_*.pkl" for e in args.envs]
        if "ood" in args.what:
            pats += [f"ood_test/{e}-medium-expert-v2/*" for e in args.envs]
        print(f"Downloading {HF_DATA_REPO} -> {DATA_DIR}")
        snapshot_download(HF_DATA_REPO, repo_type="dataset", local_dir=DATA_DIR, allow_patterns=pats, revision=args.revision)

    pats, ignore = [], []
    for e in args.envs:
        root = f"{e}_medium_expert_sparse"
        for m in args.models:
            if "density" in args.what:
                pats.append(f"{root}/neuralODE/*" if m == "neuralODE" else f"{root}/{m}_*/*")
            if "mbpo" in args.what:
                pats.append(f"{root}/mbpo_based/{m}/*")
            if "mobile" in args.what:
                pats.append(f"{root}/mobile_based/{m}/*")
    if not args.intermediate:
        ignore = ["*/checkpoints/*", "*/checkpoint_epoch_*.pt"]
    if pats:
        print(f"Downloading {HF_MODEL_REPO} -> {CKPT_DIR}")
        snapshot_download(HF_MODEL_REPO, local_dir=CKPT_DIR, allow_patterns=pats, ignore_patterns=ignore, revision=args.revision)
    print("Done.")


if __name__ == "__main__":
    main()
