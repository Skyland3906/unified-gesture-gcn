"""Clean, extensible training, evaluation, and latency benchmarking engine."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from gesture_gcn.config import TrainConfig


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class Trainer:
    """Minimal, self-contained training and evaluation loop."""

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
        )

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for features, dynamic_range, labels in loader:
                features = features.to(self.device)
                dynamic_range = dynamic_range.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(features, dynamic_range)
                loss = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        on_epoch_end: Callable[[EpochMetrics, int], None] | None = None,
    ) -> list[EpochMetrics]:
        history: list[EpochMetrics] = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            history.append(metrics)

            if on_epoch_end is not None:
                on_epoch_end(metrics, self.cfg.epochs)

            if epoch == 1 or epoch % self.cfg.log_interval == 0 or epoch == self.cfg.epochs:
                print(
                    f"[Epoch {epoch:02d}/{self.cfg.epochs:02d}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc * 100:5.1f}% | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc * 100:5.1f}%"
                )
        return history


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    in_channels: int,
    seq_len: int,
    num_joints: int,
    device: torch.device,
    warmup: int = 5,
    runs: int = 20,
) -> float:
    """Measure average single-sample inference latency in milliseconds."""
    model.eval()
    dummy_x = torch.randn(1, in_channels, seq_len, num_joints, device=device)
    dummy_dr = torch.rand(1, 3, num_joints, device=device)

    for _ in range(warmup):
        _ = model(dummy_x, dummy_dr)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(dummy_x, dummy_dr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs
    return elapsed_ms
