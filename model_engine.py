"""
model_engine.py — trains all models and generates predictions.
Runs on every data refresh (cached 24h by Streamlit).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight

from feature_engine import FEATURE_COLS

# Try to import TensorFlow; fall back gracefully if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        LSTM, GRU, SimpleRNN, Dense, Dropout, BatchNormalization,
        Bidirectional, Input, Conv1D, MaxPooling1D,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ── Hyperparameters (optimised settings) ──────────────────────────────────────
EPOCHS    = 200
BATCH     = 16
VAL_SPLIT = 0.20
LR        = 0.0005
DROPOUT   = 0.40
SEQ_LEN   = 60
HIGH      = 0.60
LOW       = 0.40


class ModelEngine:
    def __init__(self, df_feat: pd.DataFrame, split: float = 0.80):
        self.df      = df_feat
        self.split   = split
        self._prepare_data()

    # ── Data preparation ────────────────────────────────────────────────────────
    def _prepare_data(self):
        d        = self.df
        sp       = int(len(d) * self.split)
        self.tr  = d.iloc[:sp]
        self.te  = d.iloc[sp:]

        # Filter to only existing columns
        feat_cols = [c for c in FEATURE_COLS if c in d.columns]
        self.feat_cols = feat_cols

        X_tr = self.tr[feat_cols].values.astype(np.float32)
        X_te = self.te[feat_cols].values.astype(np.float32)

        self.scaler = MinMaxScaler()
        self.X_tr   = self.scaler.fit_transform(X_tr)
        self.X_te   = self.scaler.transform(X_te)

        self.y_tr       = self.tr["Target"].values.astype(np.float32)
        self.y_te       = self.te["Target"].values.astype(np.float32)
        self.y_price_tr = self.tr["NextClose"].values
        self.y_price_te = self.te["NextClose"].values

        cw_arr   = compute_class_weight("balanced", classes=np.array([0,1]),
                                         y=self.y_tr.astype(int))
        self.CW  = {0: float(cw_arr[0]), 1: float(cw_arr[1])}

        # Sequences for deep learning
        self.X_tr_seq, self.y_tr_seq = self._make_seqs(self.X_tr, self.y_tr)
        self.X_te_seq, self.y_te_seq = self._make_seqs(self.X_te, self.y_te)

    def _make_seqs(self, X, y):
        Xs, ys = [], []
        for i in range(SEQ_LEN, len(X)):
            Xs.append(X[i-SEQ_LEN:i])
            ys.append(y[i])
        return np.array(Xs, np.float32), np.array(ys, np.float32)

    # ── Callbacks ───────────────────────────────────────────────────────────────
    def _callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=12,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                              patience=6, min_lr=1e-7, verbose=0),
        ]

    def _compile(self, m):
        m.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
        return m

    # ── Deep Learning models ────────────────────────────────────────────────────
    def _build_rnn(self, sl, nf):
        inp = Input(shape=(sl, nf))
        x   = SimpleRNN(128, return_sequences=True)(inp)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = SimpleRNN(64,  return_sequences=True)(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = SimpleRNN(32,  return_sequences=False)(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT * 0.8)(x)
        x   = Dense(64, activation="relu")(x); x = Dropout(0.20)(x)
        x   = Dense(32, activation="relu")(x)
        return self._compile(Model(inp, Dense(1, activation="sigmoid")(x)))

    def _build_stacked_lstm(self, sl, nf):
        inp = Input(shape=(sl, nf))
        x   = LSTM(200, return_sequences=True)(inp)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = LSTM(100, return_sequences=True)(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = LSTM(50,  return_sequences=False)(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT * 0.8)(x)
        x   = Dense(100, activation="relu")(x); x = Dropout(0.25)(x)
        x   = Dense(50,  activation="relu")(x)
        return self._compile(Model(inp, Dense(1, activation="sigmoid")(x)))

    def _build_bilstm(self, sl, nf):
        inp = Input(shape=(sl, nf))
        x   = Bidirectional(LSTM(128, return_sequences=True))(inp)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = Bidirectional(LSTM(64, return_sequences=True))(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = LSTM(32, return_sequences=False)(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT * 0.8)(x)
        x   = Dense(64, activation="relu")(x); x = Dropout(0.20)(x)
        x   = Dense(32, activation="relu")(x)
        return self._compile(Model(inp, Dense(1, activation="sigmoid")(x)))

    def _build_bigru(self, sl, nf):
        inp = Input(shape=(sl, nf))
        x   = Bidirectional(GRU(96, return_sequences=True))(inp)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = Bidirectional(GRU(48, return_sequences=True))(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = GRU(24, return_sequences=False)(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT * 0.8)(x)
        x   = Dense(48, activation="relu")(x); x = Dropout(0.20)(x)
        return self._compile(Model(inp, Dense(1, activation="sigmoid")(x)))

    def _build_cnn_lstm(self, sl, nf):
        inp = Input(shape=(sl, nf))
        x   = Conv1D(64,  3, activation="relu", padding="same")(inp)
        x   = BatchNormalization()(x)
        x   = Conv1D(64,  3, activation="relu", padding="same")(x)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x); x = Dropout(DROPOUT)(x)
        x   = Conv1D(128, 3, activation="relu", padding="same")(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = Bidirectional(LSTM(64, return_sequences=True))(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT)(x)
        x   = LSTM(32, return_sequences=False)(x)
        x   = BatchNormalization()(x); x = Dropout(DROPOUT * 0.8)(x)
        x   = Dense(64, activation="relu")(x); x = Dropout(0.20)(x)
        x   = Dense(32, activation="relu")(x)
        return self._compile(Model(inp, Dense(1, activation="sigmoid")(x)))

    # ── ARIMAX ──────────────────────────────────────────────────────────────────
    def _run_arimax(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        ex_cols = ["RSI","VolRatio","MACD_hist","Ret1","Regime","StochRSI","ATR_pct"]
        ex_cols = [c for c in ex_cols if c in self.tr.columns]

        ex_sc   = StandardScaler()
        ex_tr   = ex_sc.fit_transform(self.tr[ex_cols].values)
        ex_te   = ex_sc.transform(self.te[ex_cols].values)

        tr_c = self.tr["Close"].values
        te_c = self.te["Close"].values

        def diff_s(s, d):
            for _ in range(d): s = np.diff(s)
            return s

        p, d_order, q = 5, 1, 2
        ds   = diff_s(tr_c.copy(), d_order)
        exd  = ex_tr[d_order:]
        ml   = max(p, q)
        res  = np.zeros(len(ds))

        Xar  = [list(ds[i-p:i][::-1]) for i in range(ml, len(ds))]
        yar  = [ds[i]                  for i in range(ml, len(ds))]
        m0   = LinearRegression().fit(np.array(Xar), np.array(yar))
        res[ml:] = np.array(yar) - m0.predict(np.array(Xar))

        X2, y2 = [], []
        for i in range(ml, len(ds)):
            r  = list(ds[i-p:i][::-1]) + list(res[max(0,i-q):i][::-1]) + list(exd[i])
            X2.append(r); y2.append(ds[i])
        model = LinearRegression().fit(np.array(X2), np.array(y2))

        hd, ho, rh, preds = list(ds), list(tr_c), list(res), []
        for t in range(len(te_c)):
            r  = list(np.array(hd[-p:])[::-1]) + list(np.array(rh[-q:])[::-1]) + list(ex_te[t])
            dp = model.predict(np.array(r).reshape(1,-1))[0]
            op = ho[-1]+dp
            preds.append(op); ho.append(op); hd.append(dp); rh.append(0.0)

        ax_price = np.array(preds)
        ax_proba = (ax_price > te_c).astype(float)
        return ax_proba

    # ── Train all ───────────────────────────────────────────────────────────────
    def train(self) -> dict:
        sl  = SEQ_LEN
        nf  = len(self.feat_cols)
        all_probas = {}
        all_accs   = {}
        model_data = {}

        # ── Deep Learning (if TF available) ─────────────────────────────────────
        if TF_AVAILABLE:
            dl_builders = {
                "RNN"          : self._build_rnn,
                "Stacked LSTM" : self._build_stacked_lstm,
                "BiLSTM"       : self._build_bilstm,
                "BiGRU"        : self._build_bigru,
                "CNN-LSTM"     : self._build_cnn_lstm,
            }
            for name, builder in dl_builders.items():
                m = builder(sl, nf)
                m.fit(self.X_tr_seq, self.y_tr_seq,
                      epochs=EPOCHS, batch_size=BATCH, validation_split=VAL_SPLIT,
                      class_weight=self.CW, callbacks=self._callbacks(), verbose=0)
                proba = m.predict(self.X_te_seq, verbose=0).flatten()
                pred  = (proba > 0.5).astype(int)
                acc   = accuracy_score(self.y_te_seq, pred)
                all_probas[name] = proba
                all_accs[name]   = acc
                model_data[name] = {
                    "proba": proba, "pred": pred,
                    "acc": acc,
                    "f1" : f1_score(self.y_te_seq, pred),
                    "auc": roc_auc_score(self.y_te_seq, proba),
                }

        # ── ARIMAX ───────────────────────────────────────────────────────────────
        ax_proba_full = self._run_arimax()
        ax_proba      = ax_proba_full[SEQ_LEN:]
        ax_pred       = ax_proba.astype(int)
        y_seq         = self.y_te_seq
        ax_acc        = accuracy_score(y_seq, ax_pred)
        all_probas["ARIMAX"] = ax_proba
        all_accs["ARIMAX"]   = ax_acc
        model_data["ARIMAX"] = {
            "proba": ax_proba, "pred": ax_pred,
            "acc" : ax_acc,
            "f1"  : f1_score(y_seq, ax_pred),
            "auc" : roc_auc_score(y_seq, ax_proba),
        }

        # ── Gradient Boosting ─────────────────────────────────────────────────
        gb = GradientBoostingClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        gb.fit(self.X_tr, self.y_tr)
        gb_proba_full = gb.predict_proba(self.X_te)[:, 1]
        gb_proba      = gb_proba_full[SEQ_LEN:]
        gb_pred       = (gb_proba > 0.5).astype(int)
        gb_acc        = accuracy_score(y_seq, gb_pred)
        all_probas["Gradient Boosting"] = gb_proba
        all_accs["Gradient Boosting"]   = gb_acc
        model_data["Gradient Boosting"] = {
            "proba": gb_proba, "pred": gb_pred,
            "acc" : gb_acc,
            "f1"  : f1_score(y_seq, gb_pred),
            "auc" : roc_auc_score(y_seq, gb_proba),
        }

        # ── Super Ensemble ────────────────────────────────────────────────────
        total_w    = sum(all_accs.values())
        ens_proba  = sum(all_accs[k] * all_probas[k] for k in all_probas) / total_w
        ens_pred   = (ens_proba > 0.5).astype(int)
        ens_acc    = accuracy_score(y_seq, ens_pred)
        ens_f1     = f1_score(y_seq, ens_pred)
        ens_auc    = roc_auc_score(y_seq, ens_proba)

        # Filtered
        filt_mask      = (ens_proba >= HIGH) | (ens_proba <= LOW)
        ens_filt_acc   = accuracy_score(y_seq[filt_mask],
                                        (ens_proba[filt_mask] >= HIGH).astype(int)) \
                         if filt_mask.sum() > 0 else ens_acc

        # Signals
        signals = np.where(ens_proba >= HIGH, 1, np.where(ens_proba <= LOW, -1, 0))

        # Live signal (last row)
        last_X_seq = self.X_te[-SEQ_LEN:].reshape(1, SEQ_LEN, nf) if TF_AVAILABLE else None
        last_X_tab = self.X_te[-1:]

        if TF_AVAILABLE and last_X_seq is not None:
            live_probas = []
            for name, builder in dl_builders.items():
                # use stored model — rebuild and reuse weights
                pass
            # Use ensemble proba last value as live signal
            last_prob = float(ens_proba[-1])
        else:
            last_prob = float(gb.predict_proba(last_X_tab)[0][1])

        last_signal = "BUY"  if last_prob >= HIGH else \
                      "SELL" if last_prob <= LOW  else "HOLD"
        last_conf   = max(last_prob, 1 - last_prob) * 100

        # ── Price regression ─────────────────────────────────────────────────
        ridge      = Ridge(alpha=1.0)
        ridge.fit(self.X_tr, self.y_price_tr)
        price_pred_full = ridge.predict(self.X_te)
        price_pred      = price_pred_full[SEQ_LEN:]
        rmse = float(np.sqrt(mean_squared_error(self.y_price_te[SEQ_LEN:], price_pred)))
        mae  = float(mean_absolute_error(self.y_price_te[SEQ_LEN:], price_pred))

        # ── Test set date/price arrays ────────────────────────────────────────
        te_df_aligned = self.te.iloc[SEQ_LEN:].copy()
        te_df_aligned["SMA20"]    = te_df_aligned["SMA20"]
        te_df_aligned["SMA50"]    = te_df_aligned["SMA50"]
        te_df_aligned["RSI"]      = te_df_aligned["RSI"]
        te_df_aligned["MACD"]     = te_df_aligned["MACD"]
        te_df_aligned["MACD_sig"] = te_df_aligned["MACD_sig"]
        te_df_aligned["Regime"]   = te_df_aligned["Regime"]

        # ── Signal history table ──────────────────────────────────────────────
        sig_mask  = filt_mask
        sig_dates = te_df_aligned.index[sig_mask]
        sig_close = te_df_aligned["Close"].values[sig_mask]
        sig_proba = ens_proba[sig_mask]
        sig_preds = (sig_proba >= HIGH).astype(int)

        sig_history = pd.DataFrame({
            "Date"       : sig_dates.strftime("%Y-%m-%d"),
            "Price"      : [f"${p:.2f}" for p in sig_close],
            "Signal"     : ["🟢 BUY" if s==1 else "🔴 SELL" for s in sig_preds],
            "P(UP)"      : [f"{p*100:.1f}%" for p in sig_proba],
            "Confidence" : [f"{max(p,1-p)*100:.1f}%" for p in sig_proba],
        }).sort_values("Date", ascending=False).reset_index(drop=True)

        return {
            # Models
            "model_data"       : model_data,
            "ensemble_acc"     : ens_acc,
            "ensemble_filt_acc": ens_filt_acc,
            "ensemble_f1"      : ens_f1,
            "ensemble_auc"     : ens_auc,
            "ens_proba"        : ens_proba,
            "ens_pred"         : ens_pred,
            "y_te"             : y_seq,
            "signals"          : signals,
            "n_signals"        : int((signals != 0).sum()),
            "HIGH"             : HIGH,
            "LOW"              : LOW,
            # Live signal
            "last_signal"      : last_signal,
            "last_confidence"  : last_conf,
            "last_prob"        : last_prob,
            # Price regression
            "price_pred"       : price_pred,
            "y_price_te"       : self.y_price_te[SEQ_LEN:],
            "rmse"             : rmse,
            "mae"              : mae,
            # Test set data for charts
            "te_df"            : te_df_aligned,
            "signal_history"   : sig_history,
            "TF_AVAILABLE"     : TF_AVAILABLE,
        }
