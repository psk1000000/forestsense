"""
ForestSense — Training Script
================================
Full training loop for the Siamese Multimodal Fusion Network.

Features:
  - CosineAnnealing LR scheduler with warm restarts
  - Early stopping (patience=10)
  - Best checkpoint saving (by val mF1)
  - CSV + per-epoch metric logging
  - Training curve auto-saving

Usage (on Colab after preprocessing):
    python train.py \
        --manifest  data/processed/manifest.csv \
        --out_dir   outputs \
        --epochs    50 \
        --batch     8 \
        --backbone  resnet34 \
        --lr        1e-4
"""

import os
import sys
import json
import csv
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# ── Project imports ────────────────────────────────────────────
# Automatically handle both local and Colab environments
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from data.dataset  import get_dataloaders
from model.architecture import build_model
from model.losses   import build_loss
from model.metrics  import ChangeDetectionMetrics, compute_batch_accuracy
from utils.visualize import plot_training_curves, save_figure


# ============================================================
#   TRAINING CONFIGURATION (can also be set via CLI)
# ============================================================

DEFAULT_CONFIG = dict(
    manifest   = "data/processed/manifest.csv",
    stats_dir  = "data/processed",
    out_dir    = "outputs",
    epochs     = 50,
    batch_size = 8,
    lr         = 3e-4,
    weight_decay = 1e-4,
    backbone   = "resnet18",
    pretrained = True,
    use_focal  = True,
    patience   = 10,
    num_workers = 2,
    amp        = True,
    seed       = 42,        # Reproducibility
)


# ============================================================
#   HELPER UTILITIES
# ============================================================

class EarlyStopping:
    """Stop training when val metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.01):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Call with current epoch's validation mF1.
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  ⏳ EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class MetricLogger:
    """Logs metrics to a CSV file for each epoch."""

    def __init__(self, path: str):
        self.path = path
        self.rows = []

    def log(self, epoch: int, phase: str, metrics: dict):
        row = {"epoch": epoch, "phase": phase, **{k: v for k, v in metrics.items()
               if not isinstance(v, dict) and k != "conf_matrix"}}
        self.rows.append(row)
        # Write immediately so it persists even if training crashes
        write_header = not Path(self.path).exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ============================================================
#   TRAIN + VALIDATE (ONE EPOCH)
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = len(loader)

    for i, batch in enumerate(loader):
        t1     = batch["t1"].to(device, non_blocking=True)
        t2     = batch["t2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(t1, t2)
            loss, ce_val, dice_val = loss_fn(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_batch_accuracy(logits.detach(), labels)

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"    Batch [{i+1:3d}/{n_batches}]  "
                  f"loss={loss.item():.4f}  "
                  f"CE={ce_val.item():.4f}  "
                  f"Dice={dice_val.item():.4f}  "
                  f"acc={total_acc/(i+1):.1f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp, num_classes=4):
    """Run validation. Returns avg loss and metric dict."""
    model.eval()
    total_loss = 0.0
    metrics    = ChangeDetectionMetrics(num_classes=num_classes)

    for batch in loader:
        t1     = batch["t1"].to(device, non_blocking=True)
        t2     = batch["t2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(t1, t2)
            loss, _, _ = loss_fn(logits, labels)

        total_loss += loss.item()
        metrics.update(logits, labels)

    results = metrics.compute()
    return total_loss / len(loader), results


# ============================================================
#   CHECKPOINT UTILITIES
# ============================================================

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer=None):
    """Load checkpoint. Returns epoch and best_score."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"✅ Checkpoint loaded from epoch {ckpt.get('epoch', '?')} "
          f"(best mF1={ckpt.get('best_mf1', '?'):.2f}%)")
    return ckpt.get("epoch", 0), ckpt.get("best_mf1", 0.0)


# ============================================================
#   MAIN TRAINING LOOP
# ============================================================

def train(cfg: dict):
    """
    Full training pipeline.

    Args:
        cfg: Configuration dict (see DEFAULT_CONFIG for keys).
    """

    # ── Reproducibility seed ─────────────────────────────────────────
    import random as _random
    import numpy as np
    seed = cfg.get("seed", 42)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"🌱 Seed: {seed}")

    # ── Setup output dirs ────────────────────────────────────
    out_dir = Path(cfg["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(out_dir / "training_log.csv")

    print("\n" + "="*60)
    print("🚀 ForestSense — Training")
    print("="*60)
    print(json.dumps({k: v for k, v in cfg.items()}, indent=2))

    # ── Device ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["amp"] and device.type == "cuda"
    print(f"\n📱 Device : {device} (AMP={use_amp})")

    # ── Data ─────────────────────────────────────────────────
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path = cfg["manifest"],
        batch_size    = cfg["batch_size"],
        num_workers   = cfg["num_workers"],
    )

    # ── Class weights ─────────────────────────────────────────
    # Load from global stats JSON
    stats_path = os.path.join(cfg["stats_dir"], "global_stats.json")
    class_weights = None
    if os.path.exists(stats_path):
        import json as _json
        with open(stats_path) as f:
            global_stats = _json.load(f)
        # Average weights across all regions
        all_w = [v["class_weights"] for v in global_stats.values()]
        avg_w = torch.tensor(
            [sum(w[i] for w in all_w) / len(all_w) for i in range(4)],
            dtype=torch.float32,
        ).to(device)
        class_weights = avg_w
        print(f"⚖️  Class weights: {avg_w.tolist()}")

    # ── Model ─────────────────────────────────────────────────
    print("\n🧠 Building model...")
    model = build_model(
        backbone  = cfg["backbone"],
        pretrained = cfg["pretrained"],
    ).to(device)

    # ── Loss & Optimiser ──────────────────────────────────────
    loss_fn = build_loss(class_weights=class_weights, use_focal=cfg["use_focal"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )
    scaler = GradScaler(enabled=use_amp)

    # ── Training state ────────────────────────────────────────
    early_stop = EarlyStopping(patience=cfg["patience"])
    logger     = MetricLogger(log_path)
    best_mf1   = 0.0
    best_ckpt  = str(ckpt_dir / "best_model.pth")

    train_losses, val_losses = [], []
    val_f1s, val_ious        = [], []

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{cfg['epochs']}  LR={scheduler.get_last_lr()[0]:.2e}")

        # Train
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, use_amp
        )
        # Validate
        val_loss, val_results = validate(model, val_loader, loss_fn, device, use_amp)

        # Scheduler step
        scheduler.step()

        elapsed = time.time() - t0
        mf1  = val_results["mF1"]
        miou = val_results["mIoU"]

        # Log
        logger.log(epoch, "train", {"loss": round(tr_loss, 5), "acc": round(tr_acc, 3)})
        logger.log(epoch, "val",   val_results)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_f1s.append(mf1)
        val_ious.append(miou)

        # Print summary
        print(f"\n  ✅ Epoch {epoch} Summary ({elapsed:.0f}s):")
        print(f"     Train Loss : {tr_loss:.4f}   Train Acc: {tr_acc:.2f}%")
        print(f"     Val Loss   : {val_loss:.4f}")
        print(f"     Val mF1    : {mf1:.2f}%   Val mIoU: {miou:.2f}%")
        print(f"     Val OA     : {val_results['OA']:.2f}%  Kappa: {val_results['kappa']:.4f}")
        print(f"     Change F1  : {val_results['change_F1']:.2f}%")

        # Save best checkpoint
        if mf1 > best_mf1:
            best_mf1 = mf1
            save_checkpoint({
                "epoch"              : epoch,
                "model_state_dict"   : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics"        : val_results,
                "config"             : cfg,
                "best_mf1"           : best_mf1,
            }, best_ckpt)

        # Save latest checkpoint (for resume)
        save_checkpoint({
            "epoch"              : epoch,
            "model_state_dict"   : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mf1"           : best_mf1,
            "config"             : cfg,
        }, str(ckpt_dir / "latest_model.pth"))

        # Early stopping
        if early_stop.step(mf1):
            print(f"\n⛔ Early stopping triggered at epoch {epoch}.")
            break

    # ── Save training curves ──────────────────────────────────
    from utils.visualize import plot_training_curves, save_figure
    fig = plot_training_curves(train_losses, val_losses, val_f1s, val_ious)
    save_figure(fig, str(out_dir / "visualizations" / "training_curves.png"))

    # ── Final report on test set ──────────────────────────────
    print("\n" + "="*60)
    print("🔍 Running FINAL EVALUATION on Test Set...")
    print("="*60)
    print(f"  Loading best checkpoint (mF1={best_mf1:.2f}%) from:\n  {best_ckpt}")
    _, _ = load_checkpoint(best_ckpt, model)
    _, test_results = validate(model, test_loader, loss_fn, device, use_amp)

    test_metrics = ChangeDetectionMetrics()
    test_metrics.conf_matrix = test_results["conf_matrix"]
    test_metrics.print_report(test_results, epoch=-1)

    # Save test metrics
    test_out = out_dir / "test_metrics.json"
    import json as _json
    with open(test_out, "w") as f:
        safe = {k: v for k, v in test_results.items() if k != "conf_matrix"}
        safe["conf_matrix"] = test_results["conf_matrix"].tolist()
        _json.dump(safe, f, indent=2)
    print(f"📝 Test metrics saved → {test_out}")
    print("\n✅ Training complete! Best model at:", best_ckpt)
    return model, test_results


# ============================================================
#   CLI ENTRY POINT
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="ForestSense Training")
    p.add_argument("--manifest",    default=DEFAULT_CONFIG["manifest"])
    p.add_argument("--stats_dir",   default=DEFAULT_CONFIG["stats_dir"])
    p.add_argument("--out_dir",     default=DEFAULT_CONFIG["out_dir"])
    p.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch",       type=int,   default=DEFAULT_CONFIG["batch_size"], dest="batch_size")
    p.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--backbone",    default=DEFAULT_CONFIG["backbone"])
    p.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    p.add_argument("--num_workers", type=int,   default=DEFAULT_CONFIG["num_workers"])
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--use_focal",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()
    cfg.update(vars(args))
    cfg["amp"] = not args.no_amp
    train(cfg)
