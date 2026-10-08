# Experiment log

All numbers below were computed by the included scripts on **procedural synthetic trajectories**, not on the actual Waymo Open Motion Dataset. These synthetic tests check numerical behavior and reproducibility, not road-driving performance.

## Runtime and repeatability

- Test environment: Linux, Python 3.13.5, PyTorch 2.10.0+cpu, no GPU.
- Deterministic train/validation/test scene generation: 192/48/48 non-overlapping scenes, seeds 42, 1,000,042, and 2,000,042.
- Data: 16 maximum agents, 64 map tokens, 11 observed states, 80 future positions per agent, 10 Hz. Agents are procedural cars, pedestrians and cyclists; their paths use linear acceleration and/or turns.
- Config: `configs/synthetic.yaml`, with a 4-epoch override for the recorded second experiment.
- Scalar metrics are computed across observed, valid agent trajectories. None of these are official Waymo metrics.

## Experiment 0 — API/learning smoke test

Command: `python scripts/smoke_test.py`

- Generates 12 linear-motion training scenes and 4 linear-motion validation scenes in a temporary directory.
- Runs 2 epochs using `configs/smoke.yaml`, computes a baseline, saves checkpoints, reloads EMA weights and checks multimodal output.
- Completed: **PASS**. Training losses 0.118768 → 0.086445 in the first version without a turn-rate prior, and validation top-1 ADE at 8 s 0.056195 → 0.035173 m.
- For exactly linear synthetic paths, constant velocity predicts perfectly (0.0 m). This is a deliberate numerical sanity check and not a meaningful learned-model benchmark.

## Experiment 1 — no physics prior (early exploratory run)

Configuration: 192 training / 48 validation synthetic mixed scenes, 48-dimensional transformer, 3 modes, 12 epochs requested; execution stopped during epoch 6; 5 completed epochs recorded.

| Epoch | Train loss | Val top-1 ADE@8s (m) | Val minADE@8s (m) |
|---|---:|---:|---:|
| 1 | 9.2728 | 5.8952 | 5.8580 |
| 2 | 8.9387 | 5.9179 | 5.7345 |
| 3 | 8.3783 | 6.0110 | 5.5335 |
| 4 | 7.9490 | 6.1360 | 5.3170 |
| 5 | 7.6758 | 6.1175 | 5.1327 |

Constant velocity on the validation set: 5.8842 m ADE@8s. The initial top-1 model **failed to improve** upon it. This led to a change in architecture: a history-only constant-turn-rate/acceleration trajectory prior, with transformer-predicted residual corrections. This is an implemented modeling revision, not a post hoc change to the evaluation labels.

## Experiment 2 — kinematic residual latent world model

To reproduce:

```bash
python scripts/generate_synthetic.py --out data/synthetic --train 192 --val 48 --test 48 --seed 42 --agents 16 --map-points 64
python scripts/train.py --config configs/synthetic.yaml --epochs 4 --train-dir data/synthetic/train --val-dir data/synthetic/val --output-dir outputs/synthetic_v2 --device cpu
python scripts/evaluate.py --checkpoint outputs/synthetic_v2/best.pt --data-dir data/synthetic/test --out outputs/synthetic_v2/test_metrics.json
```

Parameters: model width 48, 4 heads, one temporal encoder layer, 8 scene latents, one latent transformer layer, 3 future modes, dropout 0.1, AdamW learning rate 4e-4, 10-step warmup, cosine decay, weight decay 0.02, 4 epochs, batch size 16, SE(2) augmentation, gradient clipping 1, EMA 0.95.

### Validation training trajectory

| Epoch | Training loss | Val top-1 ADE@8s (m) | Val minADE@8s (m) |
|---|---:|---:|---:|
| 1 | 0.617655 | 0.433052 | 0.403507 |
| 2 | 0.522193 | 0.418969 | 0.371466 |
| 3 | 0.474974 | 0.411904 | 0.339691 |
| 4 | 0.458942 | 0.409915 | 0.324515 |

No increase in validation top-1 ADE was observed over these 4 epochs; this is **not** proof that the model would avoid overfitting with longer training or on real data.

### Separate 48-scene test split

| Method | Top-1 ADE@8s (m) | Oracle minADE@8s (m) | Top-1 endpoint within 8 m |
|---|---:|---:|---:|
| Constant velocity | 5.4038 | 5.4038 | 36.79% |
| Constant-turn/acceleration physics | 0.3269 | 0.3269 | 98.82% |
| Transformer + kinematic prior (3 modes) | 0.3607 | 0.2868 | 98.82% |

The learned model surpasses constant velocity and offers better best-of-3 predictions than the stronger physics baseline. Its *selected top-1 mode* is worse than the physics baseline in this four-epoch run. This distinction is important for reporting reliable model performance.

Full machine-readable results: `synthetic_test_metrics.json`, `synthetic_val_constant_velocity.json`, `synthetic_val_kinematic.json`, and `synthetic_train_history.csv` in this directory. The included `artifacts/synthetic_demo_best.pt` is the experiment 2 checkpoint; it is not trained on Waymo.

## Verification scope

- Unit tests cover synthetic scene shapes, masks, SE(2) rotations, gradient propagation, baseline metrics, missing-future invariance, Waymo-like feature conversion, and TFRecord CRC validation.
- End-to-end smoke test covers generated scenes, training, evaluation, saved weights, and inference.
- Waymo converter tested on manually created, schema-compatible examples. No proprietary/licensed WOMD shards were available here, so direct real-Waymo training, dataset accuracy, or official challenge metrics have **not** been validated.

## Recommended real-data experiments

1. Evaluate physics and learned models on exactly the same **official validation split** with unchanged preprocessing.
2. Train on a representative collection of real shards; examine per-class ADE/FDE, stationary objects, high-turn trajectories, intersections, crowded scenes, and horizon degradation.
3. Compare full model, no map, no neighbors, and a kinematics-only baseline under matched seeds/hardware.
4. Run multiple seeds, report means/standard deviations, plot calibration of mode confidence, and track train-vs-validation gaps.
5. Scale agent limits, map point sampling strategy, number of latent tokens, model width, and number of trajectory modes.
6. Measure collision rates, lane adherence, displacement confidence calibration, and latency. These are **not currently implemented as official metrics**.

## Final implementation regression checks

After adding the kinematic history-only prior and reinstalling the package in editable mode (`pip install -e . --no-deps --no-build-isolation`), the revised smoke test completed all 2 epochs and passed checkpoint/inference assertions. Observed loss 0.122284 → 0.087873, validation top-1 ADE@8s 0.055018 → 0.035144 m on exactly linear synthetic scenes. Constant velocity remained 0.0 m, as expected.

Final test suite: **9 passed** (all tests) with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m pytest -q`. This includes a complete synthetic TFRecord → `tf.Example` → ego-centric NPZ conversion with verified CRC.

`python -m compileall -q src scripts tests` succeeded. `scripts/visualize.py` and `scripts/predict.py` also produced `docs/demo_preview.png` and `docs/demo_prediction.json` using the included checkpoint. No CUDA performance claims, real-data results, or Docker execution claims are made.
