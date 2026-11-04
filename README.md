<h1 align="center">Generative OOD-regularized Model-based Policy Optimization (GORMPO)</h1>
<h3 align="center">NeurIPS 2026</h3>

<p align="center">
  <p align="center">
    <a href="https://aysintumay.github.io/">Aysin Tumay</a>
    ·
    <a href="https://jhhuang.site/">Jiahe Huang</a>
    ·
    Elise Jortberg
    ·
    <a href="https://roseyu.com/">Rose Yu</a>
  </p>
  <h3 align="center"><a href="https://openreview.net/forum?id=3KDoYUU2CU">Paper</a> | <a href="https://huggingface.co/aysin/GORMPO">Models</a> | <a href="https://huggingface.co/datasets/aysin/sparse_medium_expert_D4RL">Datasets</a></h3>
</p>

Welcome to the codebase for the NeurIPS 2026 paper [Generative OOD-regularized Model-based Policy Optimization](https://openreview.net/forum?id=3KDoYUU2CU)! In offline model-based RL, a policy is trained on transitions imagined by a learned world model, and those transitions can drift away from the offline data in sparse state-action spaces. GORMPO trains a generative density model on the offline dataset and uses next state-action likelihood to penalize the reward of imagined transitions that fall out of distribution. The penalty can be extracted from various density models: KDE, VAE, RealNVP, Diffusion, and Neural ODE density estimators.

This repository contains the code to train the density models, evaluate them on OOD test sets, and train GORMPO on top of two offline model-based RL algorithms, MBPO and MOBILE, on sparse medium-expert D4RL datasets (HalfCheetah, Hopper, Walker2d). It also contains the unpenalized MBPO and MOBILE baselines. 

- [Environment Setup](#environment)
- [Data and Checkpoints](#data-and-checkpoints)
- [Repository Layout](#repository-layout)
- [MBPO-based GORMPO Training](#mbpo-based-gormpo-training)
- [MOBILE-based GORMPO Training](#mobile-based-gormpo-training)
- [Training Density Models](#training-density-models)
- [OOD Testing](#ood-testing)
- [Citation](#citation)

# Setup

## Environment

The code uses Python 3.8. `mujoco-py` additionally requires the [MuJoCo 2.1 binaries](https://github.com/openai/mujoco-py#install-mujoco) under `~/.mujoco/mujoco210`.

```bash
conda create -n gormpo python=3.8
conda activate gormpo
pip install -r requirements.txt
```

Training runs are tracked with Weights & Biases; run `wandb login` before training.

## Data and Checkpoints

The sparse medium-expert D4RL datasets and OOD test sets are on [Hugging Face](https://huggingface.co/datasets/aysin/sparse_medium_expert_D4RL), and the density models and trained GORMPO runs are in [aysin/GORMPO](https://huggingface.co/aysin/GORMPO). After setting up the environment, download them into the repository:

```bash
python scripts/download_assets.py                          # everything (~3 GB)
python scripts/download_assets.py --envs hopper            # one environment
python scripts/download_assets.py --what data density      # datasets + density models only
```

Options: `--envs halfcheetah hopper walker2d`, `--models kde vae realnvp diffusion neuralODE`, `--what data ood density mbpo mobile`; `--intermediate` also fetches intermediate training checkpoints. Files keep the Hugging Face layout:

```
data/sparse_datasets/<env>_medium_expert_sparse_<p>.pkl         sparse datasets (p = 72.5, 78, 73)
data/ood_test/<env>-medium-expert-v2/ood-distance-<d>-uniform.pkl
checkpoints/<env>_medium_expert_sparse/<model>_<seed>/            density models (kde, vae, realnvp, diffusion)
checkpoints/<env>_medium_expert_sparse/neuralODE/                 Neural ODE density model
checkpoints/<env>_medium_expert_sparse/mbpo_based/<model>/seed_<seed>_*/    MBPO-based GORMPO policy + dynamics_model/
checkpoints/<env>_medium_expert_sparse/mobile_based/<model>/seed_<seed>&*/  MOBILE-based GORMPO policy + model/
```

Configs and scripts use these repo-relative paths (`data/...`, `checkpoints/...`), which are resolved against the repository root by `common/paths.py`, so they work from any folder. To keep the files elsewhere, set `GORMPO_DATA_DIR` and/or `GORMPO_CKPT_DIR`.

## Repository Layout

| Path | Contents |
|---|---|
| `density_models/` | Density estimators (KDE, VAE, RealNVP, Diffusion, Neural ODE) and their OOD tests  |
| `scripts/density_penalty.py` | Density (OOD) reward penalty used by both base algorithms |
| `common/`, `helpers/` | Buffers, loggers, evaluation and plotting utilities  |
| `configs/` | Density-model training and OOD-test configs |
| `mbpo_based/` | GORMPO on MBPO: `mopo.py`, `train.py`, `transition_model.py`, `algo/`, `models/`, `static_fns/`, `configs/`, `bash_scr/` |
| `mobile_based/` | GORMPO on MOBILE: `train.py`, `dynamics/`, `policies/`, `models/`, `utils/`, `configs/`, `config_gormpo/`, `bash_scr/` |

Both world models (`mbpo_based/transition_model.py` and `mobile_based/dynamics/ensemble_dynamics.py`) compute the penalty through `scripts.density_penalty.DensityPenalty`. Run each algorithm from its own folder; the bash scripts `cd` there automatically.

## MBPO-based GORMPO Training

Single run, from `mbpo_based/`, where `<model>` is one of `kde`, `vae`, `realnvp`, `diffusion`, `neuralODE`:

```bash
cd mbpo_based
python mopo.py --config configs/<model>/gormpo_halfcheetah_medium_expert_sparse.yaml --devid 0
```

Configs are named `mbpo_based/configs/<model>/gormpo_<env>_medium_expert_sparse.yaml` and *must* include `classifier_model_name`, `task`, `data_path` (for sparse saved datasets), `algo-name`, `reward-penalty-coef` and `penalty_type`.

Multi-seed runs (seeds 42, 123, 456) use the scripts in `mbpo_based/bash_scr/`, named `gormpo_<model>_<env>_medium_expert_sparse.sh` (and `mbpo_<env>_medium_expert_sparse.sh` for the unpenalized MBPO baseline):

```bash
bash mbpo_based/bash_scr/gormpo_kde_halfcheetah_medium_expert_sparse.sh
```

Each script writes all seeds to one CSV under `mbpo_based/results/` and computes normalized scores with `helpers/normalizer.py`.

`--dynamics-model-dir auto` (used by the scripts) loads the transition model of the downloaded paper run with the same task, density model and seed; pass a folder containing `model.pt` and `normalizers.pt` to use another one, or omit the flag to train the transition model from scratch. Trained policies are saved under `saved_models/` (`--model_path`).

## MOBILE-based GORMPO Training

Scripts in `mobile_based/bash_scr/` are named `<env>_medium_expert_sparse_<model>.sh` (no `_<model>` suffix for the plain MOBILE baseline). Each runs seeds 42, 123, 456 with the pre-trained MOBILE dynamics of the downloaded paper runs (`--dynamics_path auto`; pass a folder with `dynamics.pth`, `mu.npy`, `std.npy` to use another, or `""` to train from scratch) and appends final scores to `mobile_based/results/results_<dataset>_<model>.csv`:

```bash
bash mobile_based/bash_scr/hopper_medium_expert_sparse_kde.sh   # GORMPO-MOBILE with KDE
bash mobile_based/bash_scr/hopper_medium_expert_sparse.sh       # plain MOBILE baseline
```

Density-penalty settings come from `mobile_based/config_gormpo/<env>_medium_expert_sparse_<model>.yaml`; each script names the config it uses. 

## Training Density Models

Train a density model on the sparse dataset before running GORMPO. Run from the repository root, with `<env>` one of `halfcheetah`, `hopper`, `walker2d`. Training configs are named `configs/<model>/<env>_medium_expert_sparse.yaml`:

```bash
# KDE
python density_models/kde_module/kde.py --config configs/kde/<env>_medium_expert_sparse.yaml
# VAE
python density_models/vae_module/vae.py --config configs/vae/<env>_medium_expert_sparse.yaml
# RealNVP
python density_models/realnvp_module/realnvp.py --config configs/realnvp/<env>_medium_expert_sparse.yaml
# Neural ODE
python density_models/neuralODE/neural_ode_density.py --config configs/neuralODE/<env>_medium_expert_sparse.yaml
# Diffusion
python density_models/diffusion/ddim_training_unconditional.py \
    --config density_models/diffusion/configs/unconditional_training/<env>_medium_expert_sparse.yaml
```

Each config sets where the model is saved (`model_save_path`, or `out` for Neural ODE and Diffusion, under `checkpoints/<env>_medium_expert_sparse/`); use that path as `classifier_model_name` in the GORMPO configs. The downloaded models follow `checkpoints/<env>_medium_expert_sparse/<model>_<seed>/<model>_<seed>` (KDE, VAE, RealNVP), `.../diffusion_<seed>/checkpoint.pt` and `.../neuralODE`. The Diffusion and Neural ODE training configs read preprocessed `.npz` files that are not on Hugging Face. The MBPO multi-seed scripts contain these commands (commented out for quick policy training) for per-seed density models.

## OOD Testing

Evaluate density estimators on OOD datasets at different distance levels, i.e.,

```bash
python density_models/test_kde_ood_levels.py \
    --model_path checkpoints/halfcheetah_medium_expert_sparse/kde_42/kde_42 \
    --dataset_name halfcheetah-medium-expert-v2 \
    --sparse_dataset_name halfcheetah_medium_expert_sparse_72.5 \
    --distances 0.4 0.6 0.8 1 --suffix -uniform \
    --device cuda
```

**Outputs** (saved to `figures/{model}_ood_distance_tests/{dataset_name}/`):
- `ood_distance_summary.png`: Log-likelihood, ROC AUC, and accuracy vs distance
- `roc_curves.png`: ROC curves for each distance level
- `log_likelihood_distributions.png`: ID vs OOD distributions
- `results.json`: Numerical results for all metrics


The KDE, VAE and RealNVP scripts can also take their arguments from `configs/<model>/ood_test/<env>_medium_expert_sparse.yaml` via `--config`. For Diffusion and Neural ODE, `configs/<model>/ood_test/<env>_medium_expert_sparse_ood.yaml` configures the OOD threshold calibration in `density_models/diffusion/diffusion_ood.py` and `density_models/neuralODE/neural_ode_ood.py`.

## t-SNE Policy vs Dataset Plots

Compare policy support against the offline dataset for HalfCheetah, Hopper and Walker2d:

```bash
bash notebooks/run_tsne_policy_vs_dataset_all.sh
```

Policies are listed in `notebooks/policy_paths_{hopper,halfcheetah,walker2d}.json`; plots are saved to `results/tsne_policy_vs_dataset_sparse/{task}/`. For a single environment, run `notebooks/tsne_policy_vs_dataset.py` (see `--help`).

## Citation

If you find this repository useful for your research, please cite:

```bibtex
@inproceedings{tumay2026gormpo,
    title={Generative {OOD}-regularized Model-based Policy Optimization},
    author={Tumay, Aysin and Huang, Jiahe and Jortberg, Elise and Yu, Rose},
    booktitle={Advances in Neural Information Processing Systems},
    year={2026},
    url={https://openreview.net/forum?id=3KDoYUU2CU}
}
```

## Acknowledgement

The MBPO-based implementation in `mbpo_based/` builds on [MOPO](https://github.com/tianheyu927/mopo), and the MOBILE-based implementation in `mobile_based/` is forked from [MOBILE](https://github.com/yihaosun1124/mobile).
