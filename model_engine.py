"""
model_engine.py v3
All models implemented in pure numpy + sklearn — no TensorFlow needed.
Works on Python 3.14 / Streamlit Cloud.

Models:
  1. Vanilla RNN        (numpy forward pass + LogisticRegression head)
  2. Stacked LSTM       (numpy 2-layer LSTM + LR head)
  3. Bidirectional LSTM (numpy BiLSTM + LR head)
  4. GRU               (numpy GRU + LR head)
  5. ARIMAX            (linear AR with exogenous features)
  6. Gradient Boosting  (sklearn GradientBoostingClassifier)

All 6 combined into a super ensemble with confidence filtering.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight

from feature_engine import FEATURE_COLS

HIGH = 0.60
LOW  = 0.40
SEQ_LEN = 30   # lookback window for sequence models


# ══════════════════════════════════════════════════════════════════════════════
# NUMPY SEQUENCE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class _BaseSeqModel:
    """Base class — builds sequences, fits LR head, predicts."""

    def __init__(self, hidden=48, seed=42):
        self.H    = hidden
        self.seed = seed
        self.rng  = np.random.RandomState(seed)
        self.head = LogisticRegression(max_iter=1000, C=0.5,
                                        class_weight="balanced", random_state=seed)
        self.scaler = StandardScaler()

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _forward(self, X_seq):
        raise NotImplementedError

    def fit(self, X_seq, y):
        H = self._forward(X_seq)
        H = self.scaler.fit_transform(H)
        self.head.fit(H, y)
        return self

    def predict_proba(self, X_seq):
        H = self._forward(X_seq)
        H = self.scaler.transform(H)
        return self.head.predict_proba(H)


class VanillaRNN(_BaseSeqModel):
    """h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)"""

    def __init__(self, hidden=48, seed=42):
        super().__init__(hidden, seed)
        # Will be initialised on first forward pass when n_feat is known
        self._built = False

    def _build(self, n_feat):
        s = 0.08
        self.Wx = self.rng.randn(n_feat, self.H) * s
        self.Wh = self.rng.randn(self.H, self.H) * s
        self.b  = np.zeros(self.H)
        self._built = True

    def _forward(self, X_seq):
        n_s, sl, nf = X_seq.shape
        if not self._built: self._build(nf)
        h = np.zeros((n_s, self.H))
        for t in range(sl):
            h = np.tanh(X_seq[:, t, :] @ self.Wx + h @ self.Wh + self.b)
        return h


class LSTMNumpy(_BaseSeqModel):
    """Standard single-layer LSTM gate equations in numpy."""

    def __init__(self, hidden=48, seed=42):
        super().__init__(hidden, seed)
        self._built = False

    def _build(self, n_feat):
        s = 0.08
        d = n_feat + self.H
        self.Wi = self.rng.randn(d, self.H)*s; self.bi = np.zeros(self.H)
        self.Wf = self.rng.randn(d, self.H)*s; self.bf = np.ones(self.H)
        self.Wo = self.rng.randn(d, self.H)*s; self.bo = np.zeros(self.H)
        self.Wg = self.rng.randn(d, self.H)*s; self.bg = np.zeros(self.H)
        self._built = True

    def _step(self, x, h, c):
        xh = np.concatenate([x, h], axis=1)
        i  = self._sigmoid(xh @ self.Wi + self.bi)
        f  = self._sigmoid(xh @ self.Wf + self.bf)
        o  = self._sigmoid(xh @ self.Wo + self.bo)
        g  = np.tanh(     xh @ self.Wg + self.bg)
        c  = f * c + i * g
        h  = o * np.tanh(c)
        return h, c

    def _forward(self, X_seq):
        n_s, sl, nf = X_seq.shape
        if not self._built: self._build(nf)
        h = np.zeros((n_s, self.H)); c = np.zeros((n_s, self.H))
        for t in range(sl):
            h, c = self._step(X_seq[:, t, :], h, c)
        return h


class StackedLSTM(_BaseSeqModel):
    """Two stacked LSTM layers."""

    def __init__(self, hidden=48, seed=42):
        super().__init__(hidden, seed)
        self.lstm1 = LSTMNumpy(hidden, seed)
        self.lstm2 = LSTMNumpy(hidden, seed+1)
        self._built = False

    def _build(self, n_feat):
        self.lstm1._build(n_feat)
        self.lstm2._build(self.H)   # layer 2 input = layer 1 hidden size
        self._built = True

    def _forward(self, X_seq):
        n_s, sl, nf = X_seq.shape
        if not self._built: self._build(nf)

        # Layer 1: collect all hidden states → (n_s, sl, H)
        h1 = np.zeros((n_s, self.H)); c1 = np.zeros((n_s, self.H))
        all_h1 = []
        for t in range(sl):
            h1, c1 = self.lstm1._step(X_seq[:, t, :], h1, c1)
            all_h1.append(h1)
        seq2 = np.stack(all_h1, axis=1)  # (n_s, sl, H)

        # Layer 2: process layer-1 outputs
        h2 = np.zeros((n_s, self.H)); c2 = np.zeros((n_s, self.H))
        for t in range(sl):
            h2, c2 = self.lstm2._step(seq2[:, t, :], h2, c2)
        return h2


class BiLSTM(_BaseSeqModel):
    """Bidirectional LSTM — concat forward + backward last hidden states."""

    def __init__(self, hidden=48, seed=42):
        super().__init__(hidden, seed)
        self.fwd = LSTMNumpy(hidden, seed)
        self.bwd = LSTMNumpy(hidden, seed+7)
        self._built = False
        # Head input is 2×hidden
        self.head = LogisticRegression(max_iter=1000, C=0.5,
                                        class_weight="balanced", random_state=seed)
        self.scaler = StandardScaler()

    def _build(self, n_feat):
        self.fwd._build(n_feat)
        self.bwd._build(n_feat)
        self._built = True

    def _forward(self, X_seq):
        n_s, sl, nf = X_seq.shape
        if not self._built: self._build(nf)

        hf = np.zeros((n_s, self.H)); cf = np.zeros((n_s, self.H))
        for t in range(sl):
            hf, cf = self.fwd._step(X_seq[:, t, :], hf, cf)

        hb = np.zeros((n_s, self.H)); cb = np.zeros((n_s, self.H))
        for t in range(sl-1, -1, -1):
            hb, cb = self.bwd._step(X_seq[:, t, :], hb, cb)

        return np.concatenate([hf, hb], axis=1)  # (n_s, 2H)


class GRUNumpy(_BaseSeqModel):
    """Gated Recurrent Unit in numpy."""

    def __init__(self, hidden=48, seed=42):
        super().__init__(hidden, seed)
        self._built = False

    def _build(self, n_feat):
        s = 0.08
        d = n_feat + self.H
        self.Wr = self.rng.randn(d, self.H)*s; self.br = np.zeros(self.H)
        self.Wz = self.rng.randn(d, self.H)*s; self.bz = np.zeros(self.H)
        self.Wn = self.rng.randn(d, self.H)*s; self.bn = np.zeros(self.H)
        self._built = True

    def _forward(self, X_seq):
        n_s, sl, nf = X_seq.shape
        if not self._built: self._build(nf)
        h = np.zeros((n_s, self.H))
        for t in range(sl):
            xh = np.concatenate([X_seq[:, t, :], h], axis=1)
            r  = self._sigmoid(xh @ self.Wr + self.br)
            z  = self._sigmoid(xh @ self.Wz + self.bz)
            xrh= np.concatenate([X_seq[:, t, :], r*h], axis=1)
            n  = np.tanh(xrh @ self.Wn + self.bn)
            h  = (1 - z) * n + z * h
        return h


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ModelEngine:

    def __init__(self, df_feat: pd.DataFrame, split: float = 0.80):
        self.df    = df_feat
        self.split = split
        self._prepare_data()

    # ── Data prep ──────────────────────────────────────────────────────────────
    def _prepare_data(self):
        d  = self.df
        sp = int(len(d) * self.split)
        if sp < SEQ_LEN + 10:
            raise ValueError(f"Not enough data ({len(d)} rows). Need at least 100.")

        self.tr = d.iloc[:sp]
        self.te = d.iloc[sp:]

        feat_cols = [c for c in FEATURE_COLS if c in d.columns]
        self.feat_cols = feat_cols

        X_tr = self.tr[feat_cols].values.astype(np.float32)
        X_te = self.te[feat_cols].values.astype(np.float32)

        # Fix NaN / inf
        col_med = np.nanmedian(X_tr, axis=0)
        for arr in [X_tr, X_te]:
            bad = ~np.isfinite(arr)
            arr[bad] = np.take(col_med, np.where(bad)[1])

        self.scaler  = MinMaxScaler()
        self.X_tr    = self.scaler.fit_transform(X_tr)
        self.X_te    = self.scaler.transform(X_te)

        self.y_tr        = self.tr["Target"].values.astype(int)
        self.y_te        = self.te["Target"].values.astype(int)
        self.y_price_tr  = self.tr["NextClose"].values
        self.y_price_te  = self.te["NextClose"].values

        cw  = compute_class_weight("balanced", classes=np.array([0,1]), y=self.y_tr)
        self.CW = {0: float(cw[0]), 1: float(cw[1])}

        # Sequences for RNN/LSTM/GRU
        self.X_tr_seq, self.y_tr_seq = self._make_seqs(self.X_tr, self.y_tr)
        self.X_te_seq, self.y_te_seq = self._make_seqs(self.X_te, self.y_te)

    def _make_seqs(self, X, y):
        Xs, ys = [], []
        for i in range(SEQ_LEN, len(X)):
            Xs.append(X[i-SEQ_LEN:i])
            ys.append(y[i])
        return np.array(Xs, np.float32), np.array(ys, np.int32)

    # ── ARIMAX ─────────────────────────────────────────────────────────────────
    def _run_arimax(self):
        ex_cols = [c for c in
            ["RSI","VolRatio","MACD_hist","Ret1","Regime","ATR_pct","StochRSI"]
            if c in self.tr.columns]
        sc = StandardScaler()
        ex_tr = sc.fit_transform(self.tr[ex_cols].values)
        ex_te = sc.transform(self.te[ex_cols].values)
        tr_c  = self.tr["Close"].values
        te_c  = self.te["Close"].values

        p, d_ord, q = 5, 1, 2
        ds  = np.diff(tr_c, n=d_ord)
        exd = ex_tr[d_ord:]
        ml  = max(p, q)
        if len(ds) < ml+5:
            return np.full(len(te_c), 0.5)

        try:
            res = np.zeros(len(ds))
            Xar = [list(ds[i-p:i][::-1]) for i in range(ml, len(ds))]
            yar = list(ds[ml:])
            m0  = LinearRegression().fit(np.array(Xar), np.array(yar))
            res[ml:] = np.array(yar) - m0.predict(np.array(Xar))

            X2, y2 = [], []
            for i in range(ml, len(ds)):
                X2.append(list(ds[i-p:i][::-1]) + list(res[max(0,i-q):i][::-1]) + list(exd[i]))
                y2.append(ds[i])
            model = LinearRegression().fit(np.array(X2), np.array(y2))

            hd, ho, rh, preds = list(ds), list(tr_c), list(res), []
            for t in range(len(te_c)):
                r  = list(np.array(hd[-p:])[::-1]) + list(np.array(rh[-q:])[::-1]) + list(ex_te[t])
                dp = model.predict(np.array(r).reshape(1,-1))[0]
                op = ho[-1] + dp
                preds.append(op); ho.append(op); hd.append(dp); rh.append(0.0)

            diff   = (np.array(preds) - te_c) / (np.abs(te_c).mean() + 1e-9)
            return  1 / (1 + np.exp(-diff * 8))
        except Exception:
            return np.full(len(te_c), 0.5)

    # ── Train all ───────────────────────────────────────────────────────────────
    def train(self) -> dict:
        y_seq  = self.y_te_seq   # ground truth for sequence-window test set
        y_full = self.y_te       # full test set (for GB + ARIMAX)

        model_data  = {}
        all_probas  = {}
        all_accs    = {}

        def _eval(name, proba, y_true):
            pred = (proba > 0.5).astype(int)
            if len(np.unique(y_true)) < 2:
                acc = auc = 0.5; f1 = 0.0
            else:
                acc = accuracy_score(y_true, pred)
                f1  = f1_score(y_true, pred, zero_division=0)
                auc = roc_auc_score(y_true, proba)
            model_data[name] = {"proba":proba,"pred":pred,"acc":acc,"f1":f1,"auc":auc}
            all_probas[name] = proba
            all_accs[name]   = acc
            flag = "✅" if acc>=0.65 else "⚠️" if acc>=0.60 else "❌"
            print(f"  {flag}  {name:<22}  Acc={acc*100:.2f}%  F1={f1:.4f}")
            return acc

        print("Training sequence models (RNN / LSTM / GRU)...")

        # ── 1. Vanilla RNN ────────────────────────────────────────────────────
        try:
            rnn = VanillaRNN(hidden=48, seed=42)
            rnn.fit(self.X_tr_seq, self.y_tr_seq)
            p = rnn.predict_proba(self.X_te_seq)[:, 1]
            _eval("Vanilla RNN", p, y_seq)
        except Exception as e:
            print(f"  ❌  Vanilla RNN failed: {e}")
            all_probas["Vanilla RNN"] = np.full(len(y_seq), 0.5)
            all_accs["Vanilla RNN"]   = 0.5

        # ── 2. Stacked LSTM ───────────────────────────────────────────────────
        try:
            slstm = StackedLSTM(hidden=48, seed=42)
            slstm.fit(self.X_tr_seq, self.y_tr_seq)
            p = slstm.predict_proba(self.X_te_seq)[:, 1]
            _eval("Stacked LSTM", p, y_seq)
        except Exception as e:
            print(f"  ❌  Stacked LSTM failed: {e}")
            all_probas["Stacked LSTM"] = np.full(len(y_seq), 0.5)
            all_accs["Stacked LSTM"]   = 0.5

        # ── 3. Bidirectional LSTM ─────────────────────────────────────────────
        try:
            bilstm = BiLSTM(hidden=48, seed=42)
            bilstm.fit(self.X_tr_seq, self.y_tr_seq)
            p = bilstm.predict_proba(self.X_te_seq)[:, 1]
            _eval("Bidirectional LSTM", p, y_seq)
        except Exception as e:
            print(f"  ❌  BiLSTM failed: {e}")
            all_probas["Bidirectional LSTM"] = np.full(len(y_seq), 0.5)
            all_accs["Bidirectional LSTM"]   = 0.5

        # ── 4. GRU ────────────────────────────────────────────────────────────
        try:
            gru = GRUNumpy(hidden=48, seed=42)
            gru.fit(self.X_tr_seq, self.y_tr_seq)
            p = gru.predict_proba(self.X_te_seq)[:, 1]
            _eval("GRU", p, y_seq)
        except Exception as e:
            print(f"  ❌  GRU failed: {e}")
            all_probas["GRU"] = np.full(len(y_seq), 0.5)
            all_accs["GRU"]   = 0.5

        # ── 5. ARIMAX ─────────────────────────────────────────────────────────
        print("Training ARIMAX...")
        try:
            ax_proba_full = self._run_arimax()
            ax_proba = ax_proba_full[SEQ_LEN:]
            _eval("ARIMAX", ax_proba, y_seq)
        except Exception as e:
            print(f"  ❌  ARIMAX failed: {e}")
            all_probas["ARIMAX"] = np.full(len(y_seq), 0.5)
            all_accs["ARIMAX"]   = 0.5

        # ── 6. Gradient Boosting ──────────────────────────────────────────────
        print("Training Gradient Boosting...")
        try:
            gb = GradientBoostingClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8, random_state=42)
            gb.fit(self.X_tr, self.y_tr)
            gb_proba_full = gb.predict_proba(self.X_te)[:, 1]
            gb_proba      = gb_proba_full[SEQ_LEN:]
            _eval("Gradient Boosting", gb_proba, y_seq)
        except Exception as e:
            print(f"  ❌  GB failed: {e}")
            all_probas["Gradient Boosting"] = np.full(len(y_seq), 0.5)
            all_accs["Gradient Boosting"]   = 0.5
            gb_proba_full = np.full(len(self.y_te), 0.5)

        # ── Super Ensemble ────────────────────────────────────────────────────
        total_w   = sum(all_accs.values()) or 1.0
        ens_proba = sum(all_accs[k] * all_probas[k] for k in all_probas) / total_w
        ens_pred  = (ens_proba > 0.5).astype(int)

        if len(np.unique(y_seq)) > 1 and len(y_seq) > 0:
            ens_acc  = accuracy_score(y_seq, ens_pred)
            ens_f1   = f1_score(y_seq, ens_pred, zero_division=0)
            ens_auc  = roc_auc_score(y_seq, ens_proba)
        else:
            ens_acc = ens_f1 = ens_auc = 0.5

        filt_mask    = (ens_proba >= HIGH) | (ens_proba <= LOW)
        ens_filt_acc = (accuracy_score(y_seq[filt_mask],
                        (ens_proba[filt_mask] >= HIGH).astype(int))
                        if filt_mask.sum() > 1 and len(np.unique(y_seq[filt_mask]))>1
                        else ens_acc)

        signals   = np.where(ens_proba>=HIGH, 1, np.where(ens_proba<=LOW,-1, 0))
        last_prob = float(ens_proba[-1])
        last_sig  = "BUY" if last_prob>=HIGH else "SELL" if last_prob<=LOW else "HOLD"
        last_conf = max(last_prob, 1-last_prob)*100

        print(f"\n  🏆  Super Ensemble  Acc={ens_acc*100:.2f}%  Filtered={ens_filt_acc*100:.2f}%")

        # ── Price regression ──────────────────────────────────────────────────
        try:
            ridge = Ridge(alpha=1.0)
            ridge.fit(self.X_tr, self.y_price_tr)
            pp_full   = ridge.predict(self.X_te)
            pp_al     = pp_full[SEQ_LEN:]
            yp_al     = self.y_price_te[SEQ_LEN:]
            rmse_v    = float(np.sqrt(mean_squared_error(yp_al, pp_al)))
            mae_v     = float(mean_absolute_error(yp_al, pp_al))
        except Exception:
            pp_al  = self.te["Close"].values[SEQ_LEN:]
            yp_al  = self.y_price_te[SEQ_LEN:]
            rmse_v = mae_v = 0.0

        # ── Signal history ────────────────────────────────────────────────────
        te_al     = self.te.iloc[SEQ_LEN:].copy()
        sm        = filt_mask
        sh_dates  = te_al.index[sm]
        sh_close  = te_al["Close"].values[sm]
        sh_proba  = ens_proba[sm]
        sh_preds  = (sh_proba >= HIGH).astype(int)
        sig_hist  = pd.DataFrame({
            "Date"      : sh_dates.strftime("%Y-%m-%d"),
            "Price"     : [f"${p:.4f}" for p in sh_close],
            "Signal"    : ["🟢 BUY" if s==1 else "🔴 SELL" for s in sh_preds],
            "P(UP)"     : [f"{p*100:.1f}%" for p in sh_proba],
            "Confidence": [f"{max(p,1-p)*100:.1f}%" for p in sh_proba],
        }).sort_values("Date", ascending=False).reset_index(drop=True)

        for col in ["SMA20","SMA50","RSI","MACD","MACD_sig","Regime","BB_U","BB_L"]:
            if col not in te_al.columns:
                te_al[col] = 0.0

        return {
            "model_data"        : model_data,
            "ensemble_acc"      : ens_acc,
            "ensemble_filt_acc" : ens_filt_acc,
            "ensemble_f1"       : ens_f1,
            "ensemble_auc"      : ens_auc,
            "ens_proba"         : ens_proba,
            "ens_pred"          : ens_pred,
            "y_te"              : y_seq,
            "signals"           : signals,
            "n_signals"         : int((signals!=0).sum()),
            "HIGH"              : HIGH,
            "LOW"               : LOW,
            "last_signal"       : last_sig,
            "last_confidence"   : last_conf,
            "last_prob"         : last_prob,
            "price_pred"        : pp_al,
            "y_price_te"        : yp_al,
            "rmse"              : rmse_v,
            "mae"               : mae_v,
            "te_df"             : te_al,
            "signal_history"    : sig_hist,
            "TF_AVAILABLE"      : False,
            "SEQ_LEN"           : SEQ_LEN,
        }
