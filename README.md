# SPARK-SAM Training

## Setup

Use Linux or WSL2 with Python 3.10, CUDA-capable PyTorch 2.5.1 or newer, and the official SAM2 repository.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the additional dependencies required by the official SAM2 repository in the same environment.

Download the official SAM2.1 Tiny and Large checkpoints, then configure all paths through environment variables:

```bash
export SAM2_REPO="$PWD/external/sam2"
export SAM2_CHECKPOINT_ROOT="$PWD/checkpoints"
export ARTIFACT_ROOT="$PWD/artifacts"
export NUAA_SIRST_ROOT="$PWD/datasets/NUAA-SIRST"
export NUDT_SIRST_ROOT="$PWD/datasets/NUDT-SIRST"
export IRSTD_1K_ROOT="$PWD/datasets/IRSTD-1K"
```

`SAM2_CHECKPOINT_ROOT` must contain:

```text
sam2.1_hiera_tiny.pt
sam2.1_hiera_large.pt
```

Each dataset root must contain matching `images/` and `masks/` directories. The image and mask identifiers must match `splits/paper_split.json`.

Choose one dataset before running the commands:

```bash
export DATASET=irstd  # nuaa, nudt, or irstd
```

Run all commands from the repository root.

## Training

### 1. Train the infrared prompt estimator

```bash
python scripts/train_prompt_estimator.py \
  --config "configs/prompt_estimator/${DATASET}.yaml"
```

### 2. Build train-split response guidance

This step uses SAM2.1-Large and the trained prompt estimator.

```bash
python scripts/build_response_guidance_cache.py \
  --config "configs/response_guidance/${DATASET}.yaml" \
  --roles train \
  --device cuda
```

### 3. Build train-split dense prompt guidance

```bash
python scripts/build_prompt_guidance_cache.py \
  --config "configs/training/${DATASET}/joint_adaptation.yaml" \
  --checkpoint "artifacts/${DATASET}/prompt_estimator/train/checkpoint_best.pt" \
  --output-root "artifacts/${DATASET}/prompt_guidance" \
  --roles train \
  --device cuda
```

### 4. Run joint adaptation

```bash
python scripts/train_sparksam.py \
  --config "configs/training/${DATASET}/joint_adaptation.yaml"
```

### 5. Build the calibration response cache

```bash
python scripts/build_calibration_response_cache.py \
  --config "configs/training/${DATASET}/joint_adaptation.yaml" \
  --checkpoint "artifacts/${DATASET}/joint_adaptation/train/checkpoint_selected_sparksam.pt" \
  --output-root "artifacts/${DATASET}/calibration_response" \
  --roles train
```

### 6. Run response calibration

```bash
python scripts/train_sparksam.py \
  --config "configs/training/${DATASET}/response_calibration.yaml"
```

### 7. Run false-alarm calibration

```bash
python scripts/train_sparksam.py \
  --config "configs/training/${DATASET}/false_alarm_calibration.yaml"
```

### 8. Run high-resolution prompt refinement

```bash
python scripts/train_sparksam.py \
  --config "configs/training/${DATASET}/high_resolution_refinement.yaml"
```

The final training checkpoint is written to:

```text
artifacts/<dataset>/high_resolution_refinement/train/checkpoint_selected_sparksam.pt
```

Use `--resume auto` to continue from the latest checkpoint or `--resume none` to start a new run. Add `--nproc-per-node <gpu-count>` to `train_sparksam.py` for multi-GPU training.
