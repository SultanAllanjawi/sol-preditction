"""
model_engine.py — trains models and generates predictions.
No TensorFlow — uses ARIMAX + Gradient Boosting + Random Forest ensemble.
Robust to small datasets from CoinGecko free API.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit

from feature_engine import FEATURE_COLS

HIGH = 0.60
LOW  = 0.40


class ModelEngine:
    def __init__(self, df_feat: pd.DataFrame, split: float = 0.80):
        self.df    = df_feat
        self.split = split
        self._prepare_data()

    # ── Data preparation ────────────────────────────────────────────────────────
    def _prepare_data(self):
        d  = self.df
        sp = int(len(d) * self.split)

        # Need at least 80 rows total (64 train + 16 test)
        if len(d) < 80:
            raise ValueError(
                f"Not enough data: only {len(d)} rows. Need at least 80. "
                "Check internet connection — CoinGecko may be rate-limiting."
            )

        self.tr = d.iloc[:sp]
        self.te = d.iloc[sp:]

        # Only use columns that actually exist in the dataframe
        self.feat_cols = [c for c in FEATURE_COLS if c in d.columns]

        X_tr = self.tr[self.feat_cols].values.astype(np.float32)
        X_te = self.te[self.feat_cols].values.astype(np.float32)

        # Replace any remaining NaN/inf with column median
        for arr in [X_tr, X_te]:
            col_medians = np.nanmedian(X_tr, axis=0)
            inds = np.where(np.isnan(arr) | np.isinf(arr))
            arr[inds] = np.take(col_medians, inds[1])

        self.scaler = MinMaxScaler()
        self.X_tr   = self.scaler.fit_transform(X_tr)
        self.X_te   = self.scaler.transform(X_te)

        self.y_tr       = self.tr["Target"].values.astype(int)
        self.y_te       = self.te["Target"].values.astype(int)
        self.y_price_tr = self.tr["NextClose"].values
        self.y_price_te = self.te["NextClose"].values

        # Class weights
        unique_classes = np.unique(self.y_tr)
        if len(unique_classes) >= 2:
            cw      = compute_class_weight("balanced", classes=unique_classes, y=self.y_tr)
            self.CW = {int(k): float(v) for k, v in zip(unique_classes, cw)}
        else:
            self.CW = {0: 1.0, 1: 1.0}

    # ── ARIMAX ──────────────────────────────────────────────────────────────────
    def _run_arimax(self):
        ex_cols = [c for c in ["RSI","VolRatio","MACD_hist","Ret1","Regime","ATR_pct"]
                   if c in self.tr.columns]

        from sklearn.preprocessing import StandardScaler
        ex_sc  = StandardScaler()
        ex_tr  = ex_sc.fit_transform(self.tr[ex_cols].values)
        ex_te  = ex_sc.transform(self.te[ex_cols].values)

        tr_c = self.tr["Close"].values
        te_c = self.te["Close"].values

        def diff_s(s, d):
            for _ in range(d): s = np.diff(s)
            return s

        p, d_order, q = 3, 1, 1   # smaller order = more stable on limited data
        ds  = diff_s(tr_c.copy(), d_order)
        exd = ex_tr[d_order:]
        ml  = max(p, q)

        if len(ds) < ml + 5:
            # Not enough data for ARIMAX — return neutral 0.5
            return np.full(len(te_c), 0.5)

        res = np.zeros(len(ds))
        Xar = [list(ds[i-p:i][::-1]) for i in range(ml, len(ds))]
        yar = [ds[i]                  for i in range(ml, len(ds))]

        try:
            m0 = LinearRegression().fit(np.array(Xar), np.array(yar))
            res[ml:] = np.array(yar) - m0.predict(np.array(Xar))

            X2, y2 = [], []
            for i in range(ml, len(ds)):
                r  = list(ds[i-p:i][::-1]) + list(res[max(0,i-q):i][::-1]) + list(exd[i])
                X2.append(r); y2.append(ds[i])
            model = LinearRegression().fit(np.array(X2), np.array(y2))

            hd, ho, rh, preds = list(ds), list(tr_c), list(res), []
            for t in range(len(te_c)):
                r  = list(np.array(hd[-p:])[::-1]) + \
                     list(np.array(rh[-q:])[::-1]) + \
                     list(ex_te[t])
                dp = model.predict(np.array(r).reshape(1, -1))[0]
                op = ho[-1] + dp
                preds.append(op)
                ho.append(op); hd.append(dp); rh.append(0.0)

            ax_price = np.array(preds)
            # Convert to probability: sigmoid of normalised price difference
            diff  = (ax_price - te_c) / (np.abs(te_c).mean() + 1e-9)
            proba = 1 / (1 + np.exp(-diff * 10))   # sigmoid
            return proba

        except Exception:
            return np.full(len(te_c), 0.5)

    # ── Train all models ────────────────────────────────────────────────────────
    def train(self) -> dict:

        model_data  = {}
        all_probas  = {}
        all_accs    = {}
        y_true      = self.y_te

        # ── Gradient Boosting ──────────────────────────────────────────────────
        try:
            gb = GradientBoostingClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                random_state=42
            )
            gb.fit(self.X_tr, self.y_tr)
            gb_proba = gb.predict_proba(self.X_te)[:, 1]
            gb_pred  = (gb_proba > 0.5).astype(int)
            gb_acc   = accuracy_score(y_true, gb_pred)
            gb_f1    = f1_score(y_true, gb_pred, zero_division=0)
            gb_auc   = roc_auc_score(y_true, gb_proba) if len(np.unique(y_true)) > 1 else 0.5

            model_data["Gradient Boosting"] = {
                "proba": gb_proba, "pred": gb_pred,
                "acc": gb_acc, "f1": gb_f1, "auc": gb_auc,
            }
            all_probas["Gradient Boosting"] = gb_proba
            all_accs["Gradient Boosting"]   = gb_acc
        except Exception as e:
            # Fallback
            all_probas["Gradient Boosting"] = np.full(len(y_true), 0.5)
            all_accs["Gradient Boosting"]   = 0.5

        # ── Random Forest ──────────────────────────────────────────────────────
        try:
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=6,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            rf.fit(self.X_tr, self.y_tr)
            rf_proba = rf.predict_proba(self.X_te)[:, 1]
            rf_pred  = (rf_proba > 0.5).astype(int)
            rf_acc   = accuracy_score(y_true, rf_pred)
            rf_f1    = f1_score(y_true, rf_pred, zero_division=0)
            rf_auc   = roc_auc_score(y_true, rf_proba) if len(np.unique(y_true)) > 1 else 0.5

            model_data["Random Forest"] = {
                "proba": rf_proba, "pred": rf_pred,
                "acc": rf_acc, "f1": rf_f1, "auc": rf_auc,
            }
            all_probas["Random Forest"] = rf_proba
            all_accs["Random Forest"]   = rf_acc
        except Exception:
            all_probas["Random Forest"] = np.full(len(y_true), 0.5)
            all_accs["Random Forest"]   = 0.5

        # ── ARIMAX ─────────────────────────────────────────────────────────────
        try:
            ax_proba = self._run_arimax()
            ax_pred  = (ax_proba > 0.5).astype(int)
            ax_acc   = accuracy_score(y_true, ax_pred)
            ax_f1    = f1_score(y_true, ax_pred, zero_division=0)
            ax_auc   = roc_auc_score(y_true, ax_proba) if len(np.unique(y_true)) > 1 else 0.5

            model_data["ARIMAX"] = {
                "proba": ax_proba, "pred": ax_pred,
                "acc": ax_acc, "f1": ax_f1, "auc": ax_auc,
            }
            all_probas["ARIMAX"] = ax_proba
            all_accs["ARIMAX"]   = ax_acc
        except Exception:
            all_probas["ARIMAX"] = np.full(len(y_true), 0.5)
            all_accs["ARIMAX"]   = 0.5

        # ── Ensemble ───────────────────────────────────────────────────────────
        total_w   = sum(all_accs.values()) or 1.0
        ens_proba = sum(all_accs[k] * all_probas[k]
                        for k in all_probas) / total_w
        ens_pred  = (ens_proba > 0.5).astype(int)

        if len(np.unique(y_true)) > 1 and len(y_true) > 0:
            ens_acc  = accuracy_score(y_true, ens_pred)
            ens_f1   = f1_score(y_true, ens_pred, zero_division=0)
            ens_auc  = roc_auc_score(y_true, ens_proba)
        else:
            ens_acc = ens_f1 = ens_auc = 0.5

        # Filtered accuracy
        filt_mask = (ens_proba >= HIGH) | (ens_proba <= LOW)
        if filt_mask.sum() > 1 and len(np.unique(y_true[filt_mask])) > 1:
            ens_filt_acc = accuracy_score(
                y_true[filt_mask],
                (ens_proba[filt_mask] >= HIGH).astype(int)
            )
        else:
            ens_filt_acc = ens_acc

        # Signal arrays
        signals = np.where(ens_proba >= HIGH, 1,
                  np.where(ens_proba <= LOW,  -1, 0))

        # Live signal from last row
        last_prob   = float(ens_proba[-1])
        last_signal = "BUY"  if last_prob >= HIGH else \
                      "SELL" if last_prob <= LOW  else "HOLD"
        last_conf   = max(last_prob, 1 - last_prob) * 100

        # ── Price regression ───────────────────────────────────────────────────
        try:
            ridge      = Ridge(alpha=1.0)
            ridge.fit(self.X_tr, self.y_price_tr)
            price_pred = ridge.predict(self.X_te)
            rmse = float(np.sqrt(mean_squared_error(self.y_price_te, price_pred)))
            mae  = float(mean_absolute_error(self.y_price_te, price_pred))
        except Exception:
            price_pred = self.te["Close"].values.copy()
            rmse = mae = 0.0

        # ── Signal history ─────────────────────────────────────────────────────
        te_aligned = self.te.copy()
        sig_mask   = filt_mask
        sig_dates  = te_aligned.index[sig_mask]
        sig_close  = te_aligned["Close"].values[sig_mask]
        sig_proba  = ens_proba[sig_mask]
        sig_preds  = (sig_proba >= HIGH).astype(int)

        sig_history = pd.DataFrame({
            "Date"      : sig_dates.strftime("%Y-%m-%d"),
            "Price"     : [f"${p:.2f}"  for p in sig_close],
            "Signal"    : ["🟢 BUY" if s == 1 else "🔴 SELL" for s in sig_preds],
            "P(UP)"     : [f"{p*100:.1f}%" for p in sig_proba],
            "Confidence": [f"{max(p,1-p)*100:.1f}%" for p in sig_proba],
        }).sort_values("Date", ascending=False).reset_index(drop=True)

        # Add required columns to te_aligned if missing
        for col in ["SMA20", "SMA50", "RSI", "MACD", "MACD_sig", "Regime"]:
            if col not in te_aligned.columns:
                te_aligned[col] = 0.0

        return {
            "model_data"        : model_data,
            "ensemble_acc"      : ens_acc,
            "ensemble_filt_acc" : ens_filt_acc,
            "ensemble_f1"       : ens_f1,
            "ensemble_auc"      : ens_auc,
            "ens_proba"         : ens_proba,
            "ens_pred"          : ens_pred,
            "y_te"              : y_true,
            "signals"           : signals,
            "n_signals"         : int((signals != 0).sum()),
            "HIGH"              : HIGH,
            "LOW"               : LOW,
            "last_signal"       : last_signal,
            "last_confidence"   : last_conf,
            "last_prob"         : last_prob,
            "price_pred"        : price_pred,
            "y_price_te"        : self.y_price_te,
            "rmse"              : rmse,
            "mae"               : mae,
            "te_df"             : te_aligned,
            "signal_history"    : sig_history,
            "TF_AVAILABLE"      : False,
        }
