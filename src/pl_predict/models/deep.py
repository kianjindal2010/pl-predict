"""Layer 3: Deep sequential models — PyTorch LSTM / Transformer + neural ratings.

Predicts match outcome (H/D/A) from:
  - a window of each team's recent match features (form sequence)
  - learned team embeddings

Complements the statistical + tree layers and can be blended into the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from pl_predict.pipeline.utils import resolve_path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    _TORCH_OK = True
except Exception:  # pragma: no cover
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


@dataclass
class DeepDataset(Dataset):
    """seq_home/seq_away: (window, n_feat); home_id/away_id: team indices."""

    seqs_home: np.ndarray
    seqs_away: np.ndarray
    home_ids: np.ndarray
    away_ids: np.ndarray
    labels: np.ndarray
    match_keys: list = field(default_factory=list)
    row_ids: list = field(default_factory=list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.seqs_home[idx]).float(),
            torch.from_numpy(self.seqs_away[idx]).float(),
            torch.tensor(int(self.home_ids[idx]), dtype=torch.long),
            torch.tensor(int(self.away_ids[idx]), dtype=torch.long),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


def _pad(s: np.ndarray, window: int, n_feat: int) -> np.ndarray:
    if len(s) == 0:
        return np.zeros((window, n_feat))
    s = np.asarray(s, dtype=float)[:, :n_feat]
    if len(s) >= window:
        return s[-window:]
    pad_rows = window - len(s)
    return np.vstack([np.zeros((pad_rows, n_feat)), s])


def build_tensors(
    features: pl.DataFrame,
    window: int = 8,
    feature_cols: Optional[list[str]] = None,
    team_to_id: Optional[dict] = None,
) -> tuple[DeepDataset, dict]:
    """Convert the feature matrix into sequence tensors for the deep model.

    ``team_to_id`` may be supplied to reuse an existing team index (e.g. the
    index of a fold model). Rows referencing unknown teams are skipped.
    """
    if features.is_empty():
        raise ValueError("No features provided")
    if "target_result" not in features.columns:
        raise ValueError("target_result column required")

    feat = features.sort("date")

    # Default: home_*/away_* numeric columns
    numeric = (pl.Float64, pl.Float32, pl.Int64)
    default_cols = [
        c
        for c in feat.columns
        if c.startswith(("home_", "away_")) and feat[c].dtype in numeric
    ]
    feature_cols = feature_cols or default_cols[:20]
    if not feature_cols:
        feature_cols = [c for c in default_cols if c.startswith("home_")][:20]

    if team_to_id is None:
        teams = sorted(
            set(feat["home_team"].drop_nulls().to_list())
            | set(feat["away_team"].drop_nulls().to_list())
        )
        teams = [t for t in teams if t]
        team_to_id = {t: i for i, t in enumerate(teams)}

    h_cols = [c for c in feature_cols if c.startswith("home_")]
    a_cols = [c for c in feature_cols if c.startswith("away_")]
    if not h_cols or len(h_cols) != len(a_cols):
        # fall back to equal-length interchange
        a_cols = h_cols

    team_hist: dict = {t: [] for t in team_to_id}
    X_home, X_away, home_ids, away_ids, labels, keys, row_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    ncol = len(h_cols)

    for rec in feat.sort("date").iter_rows(named=True):
        ht = rec.get("home_team")
        at = rec.get("away_team")
        target = rec.get("target_result")
        if not ht or not at or ht not in team_to_id or at not in team_to_id:
            continue

        if target is None or (isinstance(target, float) and target != target):
            # Unplayed match: reset form history (no training signal)
            team_hist[ht] = []
            team_hist[at] = []
            continue

        home_vec = np.nan_to_num(
            [float(rec.get(c) or 0.0) for c in h_cols], nan=0.0, posinf=0.0, neginf=0.0
        )
        away_vec = np.nan_to_num(
            [float(rec.get(c) or 0.0) for c in a_cols], nan=0.0, posinf=0.0, neginf=0.0
        )
        if len(home_vec) != ncol or len(away_vec) != ncol:
            continue

        X_home.append(_pad(np.array(team_hist[ht]), window, ncol))
        X_away.append(_pad(np.array(team_hist[at]), window, ncol))
        home_ids.append(team_to_id[ht])
        away_ids.append(team_to_id[at])
        labels.append({"H": 0, "D": 1, "A": 2}[target])
        keys.append((ht, at))
        row_ids.append(rec.get("match_id"))

        team_hist[ht].append(home_vec)
        team_hist[at].append(away_vec)

    if not X_home:
        raise ValueError("No trainable samples after building sequences")

    dataset = DeepDataset(
        seqs_home=np.array(X_home),
        seqs_away=np.array(X_away),
        home_ids=np.array(home_ids, dtype=np.int64),
        away_ids=np.array(away_ids, dtype=np.int64),
        labels=np.array(labels, dtype=np.int64),
        match_keys=keys,
        row_ids=row_ids,
    )
    meta = {
        "teams": list(team_to_id.keys()),
        "team_to_id": team_to_id,
        "feature_cols": feature_cols,
        "window": window,
        "ncol": ncol,
    }
    return dataset, meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MatchSequenceModel(nn.Module):
    """Encodes home & away form sequences + team embeddings -> H/D/A logits."""

    def __init__(
        self,
        num_teams: int,
        n_features: int,
        hidden: int = 64,
        encoder: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not available")
        self.hidden = hidden
        self.encoder_type = encoder
        self.team_emb = nn.Embedding(num_teams, hidden)

        if encoder in ("lstm", "gru"):
            rnn = nn.LSTM if encoder == "lstm" else nn.GRU
            self.enc = rnn(
                n_features,
                hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        self.feed = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def _encode(self, x):
        out, _ = self.enc(x)
        return out[:, -1, :]  # last timestep representation (B, hidden)

    def forward(self, x_home, x_away, home_id, away_id):
        hh = self._encode(x_home)
        ha = self._encode(x_away)
        eh = self.team_emb(home_id)
        ea = self.team_emb(away_id)
        return self.feed(torch.cat([hh, ha, eh, ea], dim=1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _make_loader(dataset, idxs, batch_size):
    idxs = np.asarray(idxs) if idxs is not None else np.arange(len(dataset))
    sub = [dataset[int(i)] for i in idxs]
    return DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0)


def evaluate_deep(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xh, xa, h, a, y in loader:
            logits = model(xh.to(device), xa.to(device), h.to(device), a.to(device))
            pred = logits.argmax(dim=1)
            correct += (pred.cpu() == y).sum().item()
            total += len(y)
    return correct / max(total, 1)


def train_deep(
    features: pl.DataFrame,
    window: int = 8,
    encoder: str = "lstm",
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 64,
    val_frac: float = 0.15,
    device: str = "cpu",
    save_path: str = "data/processed/deep_model.pt",
    seed: int = 42,
) -> dict:
    """Train the deep model and save weights + metadata."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not available")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset, meta = build_tensors(features, window=window)
    n = len(dataset)
    idxs = np.random.RandomState(seed).permutation(n)
    n_val = int(val_frac * n)
    val_idx = idxs[:n_val]
    trn_idx = idxs[n_val:]

    train_loader = _make_loader(dataset, trn_idx, batch_size)
    val_loader = _make_loader(dataset, val_idx, batch_size)

    model = MatchSequenceModel(
        num_teams=len(meta["teams"]),
        n_features=meta["ncol"],
        hidden=hidden,
        encoder=encoder,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        tloss, tn = 0.0, 0
        for xh, xa, h, a, y in train_loader:
            xh, xa, h, a, y = (
                xh.to(device),
                xa.to(device),
                h.to(device),
                a.to(device),
                y.to(device),
            )
            opt.zero_grad()
            loss = lossf(model(xh, xa, h, a), y)
            loss.backward()
            opt.step()
            tloss += loss.item() * len(y)
            tn += len(y)

        val_acc = evaluate_deep(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(
            f"  epoch {epoch}/{epochs}  loss={tloss / max(tn, 1):.4f}  val_acc={val_acc:.3f}"
        )

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    save_path = resolve_path(save_path) if save_path else None
    if save_path is not None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": meta,
                "using": {
                    "num_teams": len(meta["teams"]),
                    "n_features": meta["ncol"],
                    "encoder": encoder,
                    "hidden": hidden,
                },
            },
            save_path,
        )
        print(f"  Deep model saved -> {save_path} (best val_acc={best_acc:.3f})")

    return {"model": model, "meta": meta, "save_path": save_path, "best_acc": best_acc}


def load_deep(model_path: str, device="cpu"):
    """Load a saved deep model, returning (model, meta).

    meta["config"] holds the model construction hyperparameters.
    """
    path = Path(resolve_path(model_path))
    ckpt = torch.load(str(path), map_location=device)
    config = ckpt["using"]
    meta = ckpt["meta"]
    meta["config"] = config
    model = MatchSequenceModel(
        num_teams=config["num_teams"],
        n_features=config["n_features"],
        hidden=config["hidden"],
        encoder=config["encoder"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta
