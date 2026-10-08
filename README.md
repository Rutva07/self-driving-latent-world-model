# Latent World Model for Autonomous Driving

**Python · PyTorch · Transformers · Waymo Open Motion Dataset (WOMD)**

A complete, runnable research starter for **ego-centric, multi-agent, multimodal future trajectory prediction**. It encodes nearby vehicles, pedestrians, cyclists and road/traffic-signal context into compact latent scene tokens, then predicts **multiple plausible 8-second futures** with confidence scores.

The repo contains: an offline synthetic simulator, a native **Waymo `tf.Example` TFRecord reader that does not require TensorFlow**, ego-centric preprocessing, a trainable latent transformer, a physics-based forecasting prior, two baselines, training with overfit safeguards, evaluation, checkpointing, visualizations, unit tests, a Docker setup, and reproducible experiments with genuine recorded outputs.

> **Evaluation status:** A synthetic-data training and test run is complete. Full Waymo GPU performance figures below are **planning targets**, not observed scores. The previously cited 85% short-horizon and 50% long-horizon accuracy figures have not been reproduced with a documented evaluation threshold. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for recorded experiments.

## Quick start (no Waymo download required)

Python 3.10+ and PyTorch 2.4+ recommended. Tested with Python 3.13.5 and PyTorch 2.10.0 CPU.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
python scripts/smoke_test.py
```

The smoke script generates temporary toy scenes, completes two train/eval epochs, saves/reloads a checkpoint, and tests finite predictions. It requires no download, internet or GPU.

**See a prediction immediately using the included trained synthetic checkpoint:**

```bash
python scripts/visualize.py \
  --checkpoint artifacts/synthetic_demo_best.pt \
  --data-dir data/demo/val --index 0 --agent 1 --out outputs/demo.png

python scripts/predict.py \
  --checkpoint artifacts/synthetic_demo_best.pt \
  --data-dir data/demo/val --index 0 --out outputs/demo_prediction.json
```

`artifacts/synthetic_demo_best.pt` was trained on procedural trajectories only; the `data/demo` scenes are also procedural, not Waymo.

## Architecture

```text
Waymo tf.Example shards / synthetic generator
                │
                ▼
     Ego-relative scene conversion
      ├─ agents [A, 11, 8]
      ├─ agent visibility masks
      ├─ map / signals [M, 6]
      └─ future labels [A, 80, 2] (TRAINING / EVALUATION ONLY)
                │
                ▼
     Historical agent transformer
        position, velocity, heading,
        dimensions, agent type, time
                │
                ├────────► Map token encoder
                │                │
                ▼                ▼
         Latent cross-attention
        L compact scene tokens
                │
         Latent transformer
                │
        Agent-to-latent decoder
                │
        ┌───────┴────────┐
        ▼                ▼
  K future modes     K mode logits
        │
        ▼
  history-only physics prior + learned residual
        │
        ▼
 [A, K, 80, 2] trajectories, [A, K] probabilities
```

The model makes simultaneous predictions for all currently visible selected road users, including the self-driving vehicle (agent 0). It **does not** simulate control decisions, feedback effects, perception from raw video/LiDAR, or physically closed-loop driving.

### Data representation and leakage prevention

WOMD uses 10 previous frames + 1 current frame and 80 future frames, all at 10 Hz. Global coordinates are in meters. The converter:

1. Finds the self-driving car (`state/is_sdc`) and transforms positions, velocities, headings, and road directions into its current local frame.
2. Selects the ego car first, then prioritizes `tracks_to_predict` flags and nearby currently valid road users.
3. Stores 11 input states per agent: `[x, y, vx, vy, sin(yaw), cos(yaw), length, width]`. Agent classes are stored separately.
4. Selects nearby roadgraph points and **current-time** traffic lights; does not read future traffic lights as features.
5. Writes padded masks and 80-frame future `(x, y)` labels. Labels are used **only** in loss/metrics, never the model forward pass.
6. Skips invalid examples, validates finite features, and keeps original train/validation/test partitions separate (do not merge them).

Default limits are `max_agents=24`, `max_map_points=128`; change these during preprocessing for a memory/coverage tradeoff. Roadgraph points are nearest-sampled, not topology-preserving polylines. Large scenes lose information under these caps.

### Core model

- **Temporal transformer:** models each agent's observed history, with visibility masks and learned time embeddings.
- **Map token encoder:** embeds ego-relative roadgraph samples and current traffic lights.
- **Perceiver-style scene compression:** learned latent queries cross-attend to agents and map points, followed by a latent transformer.
- **Agent decoder:** each agent cross-attends to a shared latent world representation.
- **Physics-residual multimodal future head:** predicts K possible paths and K probability logits. The physics prior estimates speed change and turn rate **only from observed history**; neural trajectories correct this prior.

For agent `i`, mode `k`, future step `t`:

```text
P_kin(t) = P_current + Σ_{τ <= t} 0.1 · v_estimated(τ)
P_hat_i,k(t) = P_kin_i(t) + 0.5 · t · Δ_i,k(t)
p_i,k = softmax(logit_i,k)
```

`Δ` is the learned residual and `v_estimated` is an acceleration/turn-rate rollout. Positions are meters, velocities meters/second, timestep 0.1 second. The model predicts in one batched forward pass (non-autoregressive rollout).

Training uses a **winner-take-all, masked smooth-L1 trajectory loss** (only valid future steps), an endpoint displacement auxiliary term, and a mode-probability cross-entropy term. No gradients pass through future labels into scene inputs. Agent/time padding is masked in attention and loss.

## Using the real Waymo Open Motion Dataset

WOMD data is licensed and **is not included**. Visit [Waymo's download page](https://waymo.com/open/download/) and accept its dataset terms. The current WOMD release listed there is **v1.3.1 (October 2025)**. See [the official tf.Example schema](https://waymo.com/open/data/motion/tfexample/) and [Waymo motion overview](https://waymo.com/open/data/motion/).

**Important:** Download the **Motion Dataset `tf.Example` flavor**, NOT the separate raw `Scenario` protobuf flavor. This repository's low-dependency reader supports `tf.Example` TFRecord shards directly; the two formats are not interchangeable. No TensorFlow installation is required.

After placing licensed shards in local directories, e.g. `data/waymo/train` and `data/waymo/val`, run:

```bash
python scripts/prepare_waymo.py \
  --input 'data/waymo/train/*.tfrecord*' \
  --output data/prepared/train \
  --max-agents 24 --max-map-points 128 \
  --max-scenarios 10000

python scripts/prepare_waymo.py \
  --input 'data/waymo/val/*.tfrecord*' \
  --output data/prepared/val \
  --max-agents 24 --max-map-points 128 \
  --max-scenarios 2000

python scripts/inspect_dataset.py --data-dir data/prepared/train
```

Leave `--max-scenarios` out (or set to 0) to process all shards. Use `--verify-crc` for strict (slower) checks of TFRecord checksums. The converter exports independent compressed `NPZ` files containing the seven canonical scene arrays and a `conversion.json` audit. The generated dataset and all raw licensed data are ignored by Git. Prefer a small licensed shard for the first real-data conversion test.

Potential preprocessing tradeoffs: `NPZ` per scenario can become I/O-heavy at scale; shard caches, zarr, parquet, or memory-mapped batches may be faster. Train/test metrics require ground-truth futures and therefore cannot run on the hidden-future Waymo challenge test set.

## Synthetic data and a reproducible training run

The synthetic generator does **not** approximate the full Waymo data distribution; its purpose is to make the full pipeline testable without external access. It creates mixed cars, pedestrians, cyclists, straight/turning/accelerating motion, scene masks, lane points, three independent splits and optional strictly linear trajectories.

```bash
python scripts/generate_synthetic.py --out data/synthetic \
  --train 192 --val 48 --test 48 --seed 42 \
  --agents 16 --map-points 64

python scripts/train.py --config configs/synthetic.yaml --epochs 4 \
  --train-dir data/synthetic/train --val-dir data/synthetic/val \
  --output-dir outputs/synthetic_v2

python scripts/evaluate.py \
  --checkpoint outputs/synthetic_v2/best.pt \
  --data-dir data/synthetic/test \
  --out outputs/synthetic_v2/test_metrics.json
```

The included checkpoint, CSV history, full held-out test report and experiment notes correspond to this example. You may use a GPU with `--device cuda` (or automatically with `auto`). Training defaults for full WOMD are in `configs/train.yaml`: 128-dimensional model, 16 scene latents, 6 trajectories, batch size 24, 100 epochs, EMA, and 18-epoch early stopping patience. **They are starting hyperparameters, not independently validated Waymo-optimal values.** Tune to your compute, shard count and validation metrics.

## Experimental results and extended-training targets

### Completed synthetic experiment

We trained the latent transformer with a history-only kinematic prior for **4 CPU epochs** on **192 synthetic driving scenes**, using separate **48-scene validation** and **48-scene test** sets. The synthetic generator covers straight, turning, and accelerating agents. These results test the implementation, **not** real-world Waymo forecasting performance.

| Method | Top-1 ADE over 8 s (m) ↓ | Oracle minADE over 8 s (m) ↓ | Endpoint within 8 m ↑ |
|---|---:|---:|---:|
| Constant velocity | 5.404 | 5.404 | 36.79% |
| Constant turn-rate/acceleration | **0.327** | 0.327 | **98.82%** |
| Latent transformer + physics prior (3 modes) | 0.361 | **0.287** | **98.82%** |

Our three-mode model reduced **oracle minADE by 12.2%** relative to the turn-rate/acceleration baseline. The highest-confidence mode (0.361 m ADE) did **not** beat that strong baseline (0.327 m), so improving mode ranking is an open task. Training loss decreased from **0.618 to 0.459**, and validation top-1 ADE decreased from **0.433 to 0.410 m** across the four epochs. This short run alone cannot establish generalization to Waymo.

### Expected full-scale Waymo training (targets, not measured results)

The full training configuration uses a **128-dimensional transformer**, **16 scene latents**, **6 predicted modes**, **batch size 24**, **up to 100 epochs**, mixed-precision CUDA training, EMA checkpointing, and early stopping. The following values are illustrative **engineering goals** for a successful extended run, **not** metrics from this repository's completed experiment.

| Metric | Planning range | Ambitious target |
|---|---:|---:|
| GPU training duration | 12–24 hours | 16 hours on A100 (budget) |
| Short-horizon state accuracy (1–3 s) | 75–85% | 85% |
| Long-horizon state accuracy (5–8 s) | 40–55% | 55% |
| Oracle minADE@6, 8-second horizon | 0.90–1.30 m | 0.85 m |
| Oracle minFDE@6, 8-second horizon | 1.8–2.8 m | 1.6 m |

**Interpretation:** Accuracy percentages require a fixed spatial-error threshold, target-agent population, and time aggregation before they can be evaluated. The repository's custom hit-rate thresholds are defined below; they are not interchangeable with official Waymo metrics. GPU runtime depends on scenario count, storage throughput, hardware, and early stopping. The targets above are not claims of training completed on an A100 or any other GPU.

### Time-resolved evaluation

The model observes **11 frames (1 second of history plus the current frame)** and predicts **80 future frames at 10 Hz**, enabling evaluation at **1, 3, 5, and 8 seconds**. `src/lwm/metrics.py` supports horizon-specific ADE, FDE, and hit rates; the current recorded report does **not** include separate measured short-/long-horizon accuracy values. Run the evaluator on a labeled Waymo validation split to populate these metrics.

Recorded configurations, failed and successful synthetic trials, and exact reproduction commands are in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**. Machine-readable synthetic test metrics are in `docs/synthetic_test_metrics.json`.

## Evaluation and correct terminology

The repository reports **custom open-loop metrics**, not official WOMD leaderboard scores:

| Metric | Definition |
|---|---|
| Top-1 ADE@1/3/5/8 s | Mean Euclidean position error across valid future steps, using highest-probability mode |
| Oracle minADE@1/3/5/8 s | Minimum mean error across available modes per agent; oracle selection uses labels for *evaluation only* |
| Top-1 FDE@1/3/5/8 s | Position error at the exact requested horizon, only for agents valid at that frame |
| Oracle minFDE@1/3/5/8 s | Minimum endpoint error across predicted modes |
| Top-1/oracle hit rate | Fraction within 2 m at 1 s, 4 m at 3 s, 8 m at 5/8 s (project-defined thresholds) |

Hit rate is a custom definition of "prediction accuracy". The project's **85% short-horizon / 50% long-horizon numbers remain unverified and cannot be compared** without specifying their exact dataset, agent population, horizon, confidence/mode policy and spatial thresholds.

Official Waymo motion metrics can require mAP, miss rate and challenge submission formatting, with different agent-selection and trajectory-scoring definitions. Use Waymo's official evaluator for leaderboard claims.

## Safeguards against overfitting

- Separate scene-level train/validation/test partitions; no future states as model inputs.
- Random rigid rotation augmentation of history, future labels and road tokens during training only.
- AdamW with weight decay, dropout, gradient clipping and a warmup + cosine learning-rate schedule.
- Exponential moving average of learned weights for validation and final inference.
- Early stopping on **validation top-1 ADE at 8 seconds**, checkpointing the best epoch.
- Masked losses exclude padded agents and absent future steps; no labels from future-invalid tracks enter supervised loss.
- Paired constant-velocity and turn-rate/acceleration baselines; one final held-out test report.

These techniques **reduce** overfitting risk, but cannot guarantee performance on Waymo or on rare driving behaviors. Run several seeds, plot train-vs-validation gaps and evaluate an untouched test set only after model selection.

## Commands

```bash
# Run correctness tests and end-to-end smoke test
make test
make smoke

# Convert actual licensed Waymo tf.Example shards (see above)
python scripts/prepare_waymo.py --input 'data/waymo/train/*.tfrecord*' --output data/prepared/train
python scripts/prepare_waymo.py --input 'data/waymo/val/*.tfrecord*' --output data/prepared/val

# Full Waymo training (after downloading and preprocessing)
python scripts/train.py --config configs/train.yaml \
  --train-dir data/prepared/train --val-dir data/prepared/val \
  --output-dir outputs/waymo --device auto

# Resume an interrupted training run (keep the same model config)
python scripts/train.py --config configs/train.yaml \
  --train-dir data/prepared/train --val-dir data/prepared/val \
  --output-dir outputs/waymo --resume outputs/waymo/last.pt

# Side-by-side full / no map / no neighbors ablations
python scripts/ablate.py --config configs/train.yaml \
  --train-dir data/prepared/train --val-dir data/prepared/val \
  --out outputs/ablations

# Evaluate, visualize, or export model trajectories
python scripts/evaluate.py --checkpoint outputs/waymo/best.pt --data-dir data/prepared/val
python scripts/visualize.py --checkpoint outputs/waymo/best.pt --data-dir data/prepared/val --agent 1
python scripts/predict.py --checkpoint outputs/waymo/best.pt --data-dir data/prepared/val
```

No real dataset is downloaded automatically. The CPU smoke test runs offline. GPU and long training are optional; CUDA is recommended for sizeable real-data runs.

## Docker

```bash
docker build -t latent-driving-world-model .
docker run --rm latent-driving-world-model
```

The image starts with the offline smoke test. Docker availability was **not** verified in the development environment. For training with data mounted from the host, override the container command and mount data/output volumes (and pass `--gpus all` with an appropriately configured CUDA/PyTorch image if you need GPU support). The shipped image uses standard CPU PyTorch wheels by default.

## Repository layout

```text
latent-world-model/
├── README.md
├── pyproject.toml, requirements.txt, Makefile, Dockerfile
├── configs/
│   ├── train.yaml          # full training starting configuration
│   ├── synthetic.yaml      # reproducible mixed-motion experiment
│   └── smoke.yaml          # very small offline validation
├── src/lwm/
│   ├── data/
│   │   ├── schema.py       # fixed canonical arrays and input checks
│   │   ├── synthetic.py    # procedural data generator
│   │   ├── tfrecord.py     # TFRecord / tf.Example reader + CRC
│   │   ├── waymo.py        # WOMD ego-relative conversion
│   │   └── dataset.py      # NPZ dataset + SE(2) augmentation
│   ├── models/world_model.py
│   ├── physics.py          # CV and acceleration/turn-rate rollouts
│   ├── losses.py           # masked multimodal training objective
│   ├── metrics.py          # horizon-aware ADE/FDE/hit rates
│   ├── engine.py           # training, evaluation, EMA, resume, checkpoints
│   ├── config.py
│   └── utils.py
├── scripts/
│   ├── generate_synthetic.py, prepare_waymo.py, inspect_dataset.py
│   ├── train.py, evaluate.py, ablate.py
│   ├── predict.py, visualize.py, smoke_test.py
├── tests/                  # unit and schema adapter tests
├── docs/                   # all recorded experiments and JSON metrics
├── artifacts/
│   └── synthetic_demo_best.pt
├── data/demo/              # tiny sample synthetic inference scenes
└── .github/workflows/ci.yml
```

Training saves `best.pt`, `last.pt`, `config.json`, `history.csv`, `best_metrics.json`, `baseline.json` and `kinematic_baseline.json` into the output directory. The checkpoint records optimizer and scheduler state for `--resume`. Always load checkpoints from trusted sources.

## Limitations / research next steps

- No actual licensed Waymo training or official challenge evaluation was available for this build.
- The converter supports **WOMD tf.Example**, not raw `Scenario` protobuf TFRecords, camera embeddings or LiDAR features. The newer SDC path-sample features are not consumed.
- Map sample tokens omit full polyline connectivity and high-level lane-route semantics; traffic signals are current-time-only input context.
- Fixed-size selections may omit critical far-away interaction agents; consider dynamic graph neighborhood selection and polyline encoders.
- Future trajectories are point positions, not full simulated environment state (no heading, speed, occupancy, collisions or causal closed-loop rollouts).
- Top-1 confidence calibration remains weaker than best-of-K on the tested synthetic generator.
- Add multi-seed runs, per-class metrics, calibration curves, nonlinear scenario slices, motion-challenge submission support, automated collision/lane checks and GPU throughput benchmarks before making research or safety claims.

## References

- [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)
- [Official tf.Example feature schema](https://waymo.com/open/data/motion/tfexample/)
- [Waymo Open Dataset source and evaluation code](https://github.com/waymo-research/waymo-open-dataset)
- [Waymo Dataset terms and access](https://waymo.com/open/terms/)

The synthetic generation, model code and metrics are independent educational/research implementations. This repo does not redistribute Waymo data. **Not validated for real-world autonomous-vehicle control.**
