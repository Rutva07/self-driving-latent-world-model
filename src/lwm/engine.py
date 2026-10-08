"""Trainer with AMP, EMA, checkpoint/resume, cosine schedule and early stopping."""

import contextlib
import csv
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lwm.data.dataset import SceneDataset
from lwm.losses import trajectory_loss
from lwm.metrics import MetricAccumulator, constant_velocity, physics_baseline
from lwm.models.world_model import build_model
from lwm.utils import choose_device, save_json, seed_everything, torch_load


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.detach(), 1 - self.decay)

    @contextlib.contextmanager
    def apply(self, model):
        original = {name: p.detach().clone() for name, p in model.named_parameters()}
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(original[name])


def make_loader(directory, batch_size, workers, shuffle=False, augment=False):
    ds = SceneDataset(directory, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      drop_last=False)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(model, loader, device, baseline=False, physics_mode="cv"):
    model.eval()
    collector = MetricAccumulator()
    for batch in loader:
        batch = to_device(batch, device)
        output = physics_baseline(batch, mode=physics_mode) if baseline else model(batch)
        collector.update(output, batch)
    return collector.compute()


def _write_history(path, row):
    path = Path(path)
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def _atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def train(config, train_dir, val_dir, output_dir, device_name="auto", resume=None):
    device = choose_device(device_name)
    seed_everything(config.get("seed", 42))
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    h = config["train"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_dir, h["batch_size"], h["num_workers"],
                               shuffle=True, augment=h["augment"])
    val_loader = make_loader(val_dir, h["batch_size"], h["num_workers"])
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=h["lr"],
                                  weight_decay=h["weight_decay"])
    total_updates = max(1, len(train_loader) * h["epochs"])
    warmup = min(h["warmup_steps"], max(total_updates // 10, 1))

    def lr_schedule(step):
        if step < warmup:
            return max(1e-4, (step + 1) / warmup)
        progress = min(1., (step - warmup) / max(1, total_updates - warmup))
        return h["min_lr_ratio"] + (1 - h["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    use_amp = bool(h["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, h["ema_decay"])
    start_epoch, best, bad_epochs = 0, float("inf"), 0
    if resume:
        previous = torch_load(resume, device)
        if previous["model_config"] != config["model"]:
            raise ValueError("Checkpoint model config differs from requested config")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        if previous.get("scaler"):
            scaler.load_state_dict(previous["scaler"])
        if previous.get("ema"):
            ema.shadow = {k: v.to(device) for k, v in previous["ema"].items()}
        start_epoch, best = int(previous["epoch"]) + 1, float(previous["best_metric"])
        bad_epochs = int(previous.get("bad_epochs", 0))
    baseline = evaluate_model(model, val_loader, device, baseline=True)
    save_json(output_dir / "baseline.json", baseline)
    stronger = evaluate_model(model, val_loader, device, baseline=True, physics_mode="ctrv")
    save_json(output_dir / "kinematic_baseline.json", stronger)
    save_json(output_dir / "config.json", config)
    print(f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)} ")
    print(f"constant_velocity_top1_ADE_8s={baseline.get('top1_ADE_8s_m', 'N/A')}")
    print(f"kinematic_top1_ADE_8s={stronger.get('top1_ADE_8s_m', 'N/A')}")
    best_metrics = None
    for epoch in range(start_epoch, h["epochs"]):
        model.train()
        running, updates = 0., 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{h['epochs']}", leave=False)
        for batch in progress:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch)
                loss, losses = trajectory_loss(out, batch, h["loss_confidence"], h["loss_fde"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), h["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            running += float(loss.detach())
            updates += 1
            if updates % h["log_every"] == 0:
                progress.set_postfix(loss=f"{running / updates:.3f}")
        with ema.apply(model):
            metrics = evaluate_model(model, val_loader, device)
        metric = metrics.get("top1_ADE_8s_m", float("inf"))
        improved = math.isfinite(metric) and metric < best - 1e-6
        bad_epochs = 0 if improved else bad_epochs + 1
        if improved:
            best, best_metrics = metric, metrics
            save_json(output_dir / "best_metrics.json", metrics)
        summary = {"epoch": epoch + 1, "train_loss": round(running / max(updates, 1), 6),
                   "val_top1_ADE_8s_m": round(metric, 6),
                   "val_minADE_8s_m": round(metrics.get("minADE_8s_m", float("nan")), 6),
                   "learning_rate": optimizer.param_groups[0]["lr"]}
        _write_history(output_dir / "history.csv", summary)
        print(summary)
        checkpoint = {"model": model.state_dict(), "ema": ema.shadow,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch,
                      "best_metric": best, "bad_epochs": bad_epochs,
                      "model_config": config["model"], "train_config": h}
        _atomic_checkpoint(output_dir / "last.pt", checkpoint)
        if improved:
            _atomic_checkpoint(output_dir / "best.pt", checkpoint)
        if bad_epochs >= h["patience"]:
            print(f"Early stopping after {h['patience']} unimproved epochs")
            break
    return {"best_top1_ADE_8s_m": best, "best_metrics": best_metrics,
            "baseline": baseline, "output_dir": str(output_dir)}


def load_for_evaluation(checkpoint, device="auto", use_ema=True):
    device = choose_device(device)
    saved = torch_load(checkpoint, device)
    model = build_model(saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    if use_ema and "ema" in saved:
        with torch.no_grad():
            params = dict(model.named_parameters())
            for name, value in saved["ema"].items():
                params[name].copy_(value)
    model.eval()
    return model, device, saved
