from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import ExperimentConfig
from datasets.processed import ProcessedDataset, load_processed_dataset
from evaluation.metrics import bootstrap_multitask_metrics, multitask_metrics
from models.emerge import EMERGEPredictor


class EmergeTensorDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, tensors: ProcessedDataset, indices: np.ndarray) -> None:
        self.tensors = tensors
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        idx = int(self.indices[item])
        return {
            "patient_id": str(self.tensors.patient_ids[idx]),
            "x_ehr": torch.from_numpy(self.tensors.x_ehr[idx]),
            "x_note": torch.from_numpy(self.tensors.note_embeddings[idx]),
            "x_summary": torch.from_numpy(self.tensors.summary_embeddings[idx]),
            "y_outcome": torch.tensor(self.tensors.y[idx, 0], dtype=torch.float32),
            "y_readmission": torch.tensor(self.tensors.y[idx, 1], dtype=torch.float32),
        }


def train_from_config(cfg: ExperimentConfig, seeds: tuple[int, ...] | None = None) -> pd.DataFrame:
    seeds = seeds or cfg.trainer.seeds
    rows = []
    for seed in seeds:
        run_dir = cfg.output.run_dir / f"seed_{seed}"
        result = train_one_seed(cfg, seed=seed, output_dir=run_dir)
        rows.append(
            {
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "monitor": result["monitor"],
                "best_val_score": result["best_val_score"],
                **{
                    key: value
                    for key, value in result["test_metrics"].items()  # type: ignore[union-attr]
                },
            }
        )
    summary = pd.DataFrame(rows)
    cfg.output.run_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg.output.run_dir / "summary_by_seed.csv", index=False)
    _write_summary_json(summary, cfg.output.run_dir / "summary.json")
    return summary


def train_one_seed(cfg: ExperimentConfig, *, seed: int, output_dir: Path) -> dict[str, object]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = load_processed_dataset(cfg.data.processed_dir)
    device = _resolve_device(cfg.trainer.device)
    loaders = _build_loaders(cfg, tensors)
    model = EMERGEPredictor(
        input_ehr_dim=int(tensors.metadata["input_ehr_dim"]),
        input_note_dim=int(tensors.metadata["input_note_dim"]),
        input_summary_dim=int(tensors.metadata["input_summary_dim"]),
        hidden_dim=cfg.model.hidden_dim,
        ehr_net=cfg.model.ehr_net,
        text_fusion=cfg.model.text_fusion,
        modality_fusion=cfg.model.modality_fusion,
        use_modality=cfg.model.use_modality,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    monitor = _monitor_metric(cfg.trainer.focus_task)
    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    best_path = output_dir / "best.pt"

    for epoch in range(cfg.trainer.max_epochs):
        train_loss = _train_epoch(model, loaders["train"], optimizer, loss_fn, cfg, device)
        val_loss, val_preds = _evaluate(model, loaders["val"], loss_fn, cfg, device)
        val_metrics = multitask_metrics(**_metric_inputs(val_preds))
        score = val_metrics.get(monitor, float("nan"))
        if np.isnan(score):
            score = -float("inf")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        print(
            f"seed={seed} epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} {monitor}={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": _jsonable(asdict(cfg)),
                    "metadata": tensors.metadata,
                    "seed": seed,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.trainer.patience:
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_preds = _evaluate(model, loaders["test"], loss_fn, cfg, device)
    test_metrics = multitask_metrics(**_metric_inputs(test_preds))
    bootstrap = bootstrap_multitask_metrics(**_metric_inputs(test_preds), rounds=10, seed=seed)
    predictions = pd.DataFrame(
        {
            "patient_id": test_preds["patient_ids"],
            "outcome_true": test_preds["outcome_true"],
            "outcome_pred": test_preds["outcome_pred"],
            "readmission_true": test_preds["readmission_true"],
            "readmission_pred": test_preds["readmission_pred"],
        }
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    result = {
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "monitor": monitor,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "bootstrap": bootstrap,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_loaders(cfg: ExperimentConfig, tensors: ProcessedDataset) -> dict[str, DataLoader]:
    datasets = {
        "train": EmergeTensorDataset(tensors, tensors.train_idx),
        "val": EmergeTensorDataset(tensors, tensors.val_idx),
        "test": EmergeTensorDataset(tensors, tensors.test_idx),
    }
    return {
        "train": DataLoader(datasets["train"], batch_size=cfg.trainer.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=cfg.trainer.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=cfg.trainer.batch_size, shuffle=False),
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, batch, device)
        loss = _loss(logits, batch, loss_fn, cfg, device)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[float, dict[str, np.ndarray | list[str]]]:
    model.eval()
    losses = []
    patient_ids: list[str] = []
    outcome_pred: list[np.ndarray] = []
    readmission_pred: list[np.ndarray] = []
    outcome_true: list[np.ndarray] = []
    readmission_true: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = _forward(model, batch, device)
            loss = _loss(logits, batch, loss_fn, cfg, device)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            losses.append(float(loss.detach().cpu()))
            patient_ids.extend(batch["patient_id"])
            outcome_pred.append(probs[:, 0])
            readmission_pred.append(probs[:, 1])
            outcome_true.append(batch["y_outcome"].numpy())
            readmission_true.append(batch["y_readmission"].numpy())
    return float(np.mean(losses)) if losses else 0.0, {
        "patient_ids": patient_ids,
        "outcome_pred": np.concatenate(outcome_pred) if outcome_pred else np.array([]),
        "readmission_pred": np.concatenate(readmission_pred) if readmission_pred else np.array([]),
        "outcome_true": np.concatenate(outcome_true) if outcome_true else np.array([]),
        "readmission_true": np.concatenate(readmission_true) if readmission_true else np.array([]),
    }


def _forward(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return model(
        batch["x_ehr"].to(device),
        batch["x_note"].to(device),
        batch["x_summary"].to(device),
    )


def _loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    cfg: ExperimentConfig,
    device: torch.device,
) -> torch.Tensor:
    outcome = batch["y_outcome"].to(device)
    readmission = batch["y_readmission"].to(device)
    loss_outcome = loss_fn(logits[:, 0], outcome)
    loss_readmission = loss_fn(logits[:, 1], readmission)
    if cfg.trainer.focus_task == "outcome":
        return loss_outcome
    if cfg.trainer.focus_task == "readmission":
        return loss_readmission
    if cfg.trainer.focus_task == "multitask":
        return loss_outcome + loss_readmission
    raise ValueError(f"Unsupported focus task: {cfg.trainer.focus_task}")


def _monitor_metric(focus_task: str) -> str:
    if focus_task == "readmission":
        return "readmission_auprc"
    return "outcome_auprc"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _metric_inputs(preds: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
    return {
        "outcome_pred": preds["outcome_pred"],  # type: ignore[dict-item]
        "readmission_pred": preds["readmission_pred"],  # type: ignore[dict-item]
        "outcome_true": preds["outcome_true"],  # type: ignore[dict-item]
        "readmission_true": preds["readmission_true"],  # type: ignore[dict-item]
    }


def _write_summary_json(summary: pd.DataFrame, path: Path) -> None:
    metric_cols = [
        col
        for col in summary.columns
        if col
        not in {
            "seed",
            "best_epoch",
            "monitor",
            "best_val_score",
        }
    ]
    payload = {
        "n_seeds": int(len(summary)),
        "metrics": {
            col: {
                "mean": float(summary[col].mean()),
                "std": float(summary[col].std(ddof=0)) if len(summary) > 1 else 0.0,
            }
            for col in metric_cols
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
