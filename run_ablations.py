"""Run lambda sweep + no-confidence ablation; write ablation_results.json."""
import json
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 42
CSV_PATH = "assistments_2009.csv"
LAMBDA_CAL = 0.3
MAX_EPOCHS = 40
PATIENCE = 3

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, flush=True)


def detect_column(df, candidates, description):
    norm = {c: re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns}
    cand_norm = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in candidates]
    for want in cand_norm:
        for col, n in norm.items():
            if n == want or n.endswith("_" + want) or want in n.split("_"):
                return col
    raise ValueError(description)


def load_data():
    df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
    col_user = detect_column(df, ["user_id", "userid", "student_id", "user"], "user_id")
    col_skill_name = detect_column(df, ["skill_name", "skillname", "kc", "skill"], "skill_name")
    col_correct = detect_column(df, ["correct", "is_correct", "label"], "correct")
    col_order = detect_column(df, ["order_id", "orderid", "log_id", "row_id"], "order_id")

    work = df[[col_user, col_skill_name, col_correct, col_order]].copy()
    work.columns = ["user_id", "skill_name", "correct", "order_id"]
    work["correct"] = pd.to_numeric(work["correct"], errors="coerce")
    work = work.dropna(subset=["correct"])
    counts = work.groupby("user_id").size()
    work = work[work["user_id"].isin(counts[counts >= 10].index)].copy()
    work["order_id"] = pd.to_numeric(work["order_id"], errors="coerce")
    work = work.dropna(subset=["order_id", "skill_name"])
    work["skill_name"] = work["skill_name"].astype(str)
    work["correct"] = work["correct"].astype(np.float32).clip(0, 1)

    time_col = None
    for c in df.columns:
        n = re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        if n in ("ms_first_response", "msfirstresponse", "response_time", "duration_ms"):
            time_col = c
            break
    if time_col:
        work["response_time_ms"] = pd.to_numeric(df.loc[work.index, time_col], errors="coerce")
    else:
        work["response_time_ms"] = np.nan

    work = work.sort_values(["user_id", "order_id"], kind="mergesort").reset_index(drop=True)
    skill_codes, _ = pd.factorize(work["skill_name"], sort=True)
    work["skill_id"] = skill_codes.astype(np.int64)
    num_skills = int(work["skill_id"].max()) + 1

    if work["response_time_ms"].notna().any():
        med = work["response_time_ms"].median()
        iqr = work["response_time_ms"].quantile(0.75) - work["response_time_ms"].quantile(0.25)
        scale = float(iqr) if iqr and iqr > 0 else float(work["response_time_ms"].std() or 1.0)
        work["response_time_norm"] = ((work["response_time_ms"] - med) / scale).clip(-5, 5).fillna(0.0)
    else:
        work["response_time_norm"] = 0.0

    rng = np.random.default_rng(RANDOM_SEED)
    correct_mask = work["correct"].to_numpy() >= 0.5
    conf = np.empty(len(work), dtype=np.float32)
    conf[correct_mask] = rng.beta(8, 2, size=int(correct_mask.sum()))
    conf[~correct_mask] = rng.beta(2, 8, size=int((~correct_mask).sum()))
    work["confidence"] = np.clip(conf + rng.normal(0, 0.05, len(work)), 0.0, 1.0)

    all_users = work["user_id"].unique()
    train_users, temp_users = train_test_split(all_users, test_size=0.2, random_state=RANDOM_SEED)
    val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=RANDOM_SEED)
    train_df = work[work["user_id"].isin(train_users)]
    val_df = work[work["user_id"].isin(val_users)]
    test_df = work[work["user_id"].isin(test_users)]
    return work, train_df, val_df, test_df, num_skills


def sequences_from_df(frame, num_skills):
    seqs = []
    for _, g in frame.groupby("user_id", sort=False):
        g = g.sort_values("order_id", kind="mergesort")
        s = g["skill_id"].to_numpy(np.int64)
        y = g["correct"].to_numpy(np.float32)
        c = g["confidence"].to_numpy(np.float32)
        rt = g["response_time_norm"].to_numpy(np.float32)
        if len(s) > 0:
            seqs.append((s, y, c, rt))
    return seqs


class DKTSimpleDataset(Dataset):
    def __init__(self, seqs, num_skills, zero_confidence=False):
        self.seqs = [(s, y, c, rt) for s, y, c, rt in seqs if len(s) >= 2]
        self.K = num_skills
        self.zero_confidence = zero_confidence

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s, y, c, rt = self.seqs[idx]
        T = len(s)
        T_align = T - 1
        x = np.zeros((T_align, 2 * self.K + 2), dtype=np.float32)
        for t in range(T_align):
            k = int(s[t])
            x[t, k] = 1.0
            x[t, self.K + k] = float(y[t])
            x[t, 2 * self.K] = 0.0 if self.zero_confidence else float(c[t])
            x[t, 2 * self.K + 1] = float(rt[t])
        y_kt = y[1:T].astype(np.float32, copy=True)
        y_cal = (c[:T_align] - y[:T_align]).astype(np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y_kt), torch.from_numpy(y_cal), T_align


def collate_pad(batch):
    xs, y_kt, y_cal, lens = zip(*batch)
    B = len(batch)
    Tm = max(lens)
    D = xs[0].shape[1]
    xb = torch.zeros(B, Tm, D, dtype=torch.float32)
    y_kt_b = torch.zeros(B, Tm, dtype=torch.float32)
    y_cal_b = torch.zeros(B, Tm, dtype=torch.float32)
    mask = torch.zeros(B, Tm, dtype=torch.bool)
    for i, (x, yk, yc, L) in enumerate(batch):
        xb[i, :L] = x
        y_kt_b[i, :L] = yk
        y_cal_b[i, :L] = yc
        mask[i, :L] = True
    return xb, y_kt_b, y_cal_b, mask


class JointKTCalibrationModel(nn.Module):
    def __init__(self, num_skills, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(2 * num_skills + 2, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.kt_head = nn.Linear(hidden_size, 1)
        self.cal_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        pred_kt = torch.sigmoid(self.kt_head(h).squeeze(-1))
        pred_cal = torch.tanh(self.cal_head(h).squeeze(-1))
        return pred_kt, pred_cal


kt_criterion = nn.BCELoss(reduction="none")
cal_criterion = nn.MSELoss(reduction="none")


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        prop = np.mean(in_bin)
        if prop > 0:
            ece += prop * abs(np.mean(y_true[in_bin]) - np.mean(y_prob[in_bin]))
    return float(ece)


def run_joint_epoch(model, loader, optimizer, train_mode, lambda_cal=0.3):
    model.train() if train_mode else model.eval()
    total_kt, total_cal, ntok = 0.0, 0.0, 0
    for xb, y_kt, y_cal, mask in loader:
        xb, y_kt, y_cal, mask = xb.to(DEVICE), y_kt.to(DEVICE), y_cal.to(DEVICE), mask.to(DEVICE)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            pred_kt, pred_cal = model(xb)
            m = mask.float()
            denom = m.sum().clamp_min(1.0)
            kt_loss = (kt_criterion(pred_kt, y_kt) * m).sum() / denom
            cal_loss = (cal_criterion(pred_cal, y_cal) * m).sum() / denom
            loss = kt_loss + lambda_cal * cal_loss
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        n = int(mask.sum().item())
        total_kt += float(kt_loss.item()) * n
        total_cal += float(cal_loss.item()) * n
        ntok += n
    return total_kt / max(ntok, 1), total_cal / max(ntok, 1)


def train_joint_model(model, train_loader, val_loader, lambda_cal=0.3, label="Joint Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_val_kt_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_kt, tr_cal = run_joint_epoch(model, train_loader, optimizer, True, lambda_cal)
        va_kt, va_cal = run_joint_epoch(model, val_loader, optimizer, False, lambda_cal)
        scheduler.step(va_kt)
        if va_kt < best_val_kt_loss:
            best_val_kt_loss = va_kt
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"[{label}] Epoch {epoch}/{MAX_EPOCHS}  Val KT: {va_kt:.4f}", flush=True)
        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping after {epoch} epochs.", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate_joint(model, loader):
    model.eval()
    all_y_kt, all_p_kt, all_y_cal, all_p_cal = [], [], [], []
    with torch.no_grad():
        for xb, y_kt, y_cal, mask in loader:
            xb = xb.to(DEVICE)
            pred_kt, pred_cal = model(xb)
            pred_kt = pred_kt.cpu().numpy()
            pred_cal = pred_cal.cpu().numpy()
            m = mask.numpy()
            all_y_kt.append(y_kt.numpy()[m])
            all_p_kt.append(pred_kt[m])
            all_y_cal.append(y_cal.numpy()[m])
            all_p_cal.append(pred_cal[m])
    y_kt_true = np.concatenate(all_y_kt).astype(np.float64)
    y_kt_prob = np.concatenate(all_p_kt).astype(np.float64)
    y_cal_true = np.concatenate(all_y_cal).astype(np.float64)
    y_cal_pred = np.concatenate(all_p_cal).astype(np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_kt_true, y_kt_prob)))
    cal_mse = float(mean_squared_error(y_cal_true, y_cal_pred))
    auc = float(roc_auc_score(y_kt_true, y_kt_prob)) if len(np.unique(y_kt_true)) >= 2 else float("nan")
    ece = compute_ece(y_kt_true, y_kt_prob)
    return {"auc": auc, "rmse": rmse, "cal_mse": cal_mse, "ece": ece}


def make_loaders(train_seqs, val_seqs, test_seqs, num_skills, zero_confidence=False):
    kw = {"zero_confidence": zero_confidence}
    train_loader = DataLoader(DKTSimpleDataset(train_seqs, num_skills, **kw), batch_size=32, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(DKTSimpleDataset(val_seqs, num_skills, **kw), batch_size=64, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(DKTSimpleDataset(test_seqs, num_skills, **kw), batch_size=64, shuffle=False, collate_fn=collate_pad)
    return train_loader, val_loader, test_loader


def main():
    _, train_df, val_df, test_df, num_skills = load_data()
    train_seqs = sequences_from_df(train_df, num_skills)
    val_seqs = sequences_from_df(val_df, num_skills)
    test_seqs = sequences_from_df(test_df, num_skills)
    train_loader, val_loader, test_loader = make_loaders(train_seqs, val_seqs, test_seqs, num_skills)

    results = {"lambda_sweep": {}, "noconf": None, "full": None}

    for lam in [0.1, 0.3, 0.5]:
        print(f"\n=== Lambda={lam} ===", flush=True)
        lam_model = JointKTCalibrationModel(num_skills).to(DEVICE)
        train_joint_model(lam_model, train_loader, val_loader, lambda_cal=lam, label=f"Lambda={lam}")
        results["lambda_sweep"][str(lam)] = evaluate_joint(lam_model, test_loader)
        print(f"Lambda={lam} metrics:", results["lambda_sweep"][str(lam)], flush=True)

    print("\n=== No-Confidence Ablation ===", flush=True)
    nc_train, nc_val, nc_test = make_loaders(train_seqs, val_seqs, test_seqs, num_skills, zero_confidence=True)
    noconf_model = JointKTCalibrationModel(num_skills).to(DEVICE)
    train_joint_model(noconf_model, nc_train, nc_val, lambda_cal=LAMBDA_CAL, label="No Confidence")
    results["noconf"] = evaluate_joint(noconf_model, nc_test)
    print("No-conf metrics:", results["noconf"], flush=True)

    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote ablation_results.json", flush=True)


if __name__ == "__main__":
    main()
