from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import MODEL_DIR

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # PyTorch is optional; the caller will fall back automatically.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


MODEL_PATH = Path(MODEL_DIR) / "cnn_bilstm_demand.pt"
SCALER_PATH = Path(MODEL_DIR) / "cnn_bilstm_scaler.pkl"


class CNNBiLSTMModel(nn.Module if nn is not None else object):
    """Small CPU-friendly CNN-BiLSTM for grid-level short-term demand prediction."""

    def __init__(self, feature_size: int, hidden_size: int = 32):
        if nn is None:
            raise RuntimeError("PyTorch is not installed.")
        super().__init__()
        self.conv = nn.Conv1d(feature_size, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden_size * 2, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        # x: [batch, window, features]
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def torch_available() -> bool:
    return torch is not None


def build_sequence_dataset(features: pd.DataFrame, feature_columns: list[str], window: int = 6) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build rolling 6-hour sequences per grid and next-hour start/end targets."""
    if features.empty:
        return np.empty((0, window, len(feature_columns))), np.empty((0, 2)), pd.DataFrame()

    rows_x: list[np.ndarray] = []
    rows_y: list[list[float]] = []
    meta_rows: list[dict[str, Any]] = []
    df = features.sort_values(["grid_id", "stat_time"]).copy()

    for grid_id, group in df.groupby("grid_id"):
        group = group.sort_values("stat_time").reset_index(drop=True)
        if len(group) <= window:
            continue
        values = group[feature_columns].fillna(0).astype(float).values
        targets = group[["target_start_count", "target_end_count"]].astype(float).values
        for idx in range(window, len(group)):
            if np.isnan(targets[idx]).any():
                continue
            rows_x.append(values[idx - window : idx])
            rows_y.append([max(targets[idx][0], 0), max(targets[idx][1], 0)])
            meta_rows.append({"grid_id": grid_id, "stat_time": group.loc[idx, "stat_time"]})

    if not rows_x:
        return np.empty((0, window, len(feature_columns))), np.empty((0, 2)), pd.DataFrame()
    return np.asarray(rows_x, dtype=np.float32), np.asarray(rows_y, dtype=np.float32), pd.DataFrame(meta_rows)


def train_cnn_bilstm(
    features: pd.DataFrame,
    feature_columns: list[str],
    window: int = 6,
    epochs: int = 35,
    batch_size: int = 64,
) -> dict:
    if torch is None:
        return {"success": False, "message": "当前环境未安装 PyTorch，系统已自动使用轻量预测模型。"}

    x, y, _ = build_sequence_dataset(features, feature_columns, window=window)
    if len(x) < 80:
        return {"success": False, "message": "有效序列样本不足，跳过 CNN-BiLSTM。"}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    flat = x.reshape(-1, x.shape[-1])
    scaled = scaler.fit_transform(flat).reshape(x.shape).astype(np.float32)

    split = max(int(len(scaled) * 0.8), 1)
    x_train, y_train = scaled[:split], y[:split]
    x_test, y_test = scaled[split:], y[split:]
    if len(x_test) == 0:
        x_test, y_test = x_train, y_train

    device = torch.device("cpu")
    model = CNNBiLSTMModel(feature_size=len(feature_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_test).to(device)).cpu().numpy()
    pred = np.maximum(pred, 0)
    mae = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    total_var = float(np.sum((y_test - np.mean(y_test, axis=0)) ** 2))
    r2 = None if total_var == 0 else float(1 - np.sum((y_test - pred) ** 2) / total_var)

    torch.save({"model_state": model.state_dict(), "feature_size": len(feature_columns), "feature_columns": feature_columns}, MODEL_PATH)
    joblib.dump({"scaler": scaler, "window": window, "feature_columns": feature_columns}, SCALER_PATH)
    return {
        "success": True,
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "sequence_count": int(len(x)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def load_model(model_path: Path = MODEL_PATH):
    if torch is None or not model_path.exists():
        return None
    payload = torch.load(model_path, map_location="cpu")
    model = CNNBiLSTMModel(feature_size=int(payload["feature_size"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def predict_with_cnn_bilstm(latest_sequences: np.ndarray, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("CNN-BiLSTM model or scaler file does not exist.")
    model = load_model(model_path)
    scaler_payload = joblib.load(scaler_path)
    scaler: StandardScaler = scaler_payload["scaler"]
    scaled = scaler.transform(latest_sequences.reshape(-1, latest_sequences.shape[-1])).reshape(latest_sequences.shape).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.tensor(scaled)).cpu().numpy()
    return np.maximum(pred, 0)


def save_model(*_args, **_kwargs) -> None:
    """Kept for explicit API completeness; training already persists model and scaler."""
    return None
