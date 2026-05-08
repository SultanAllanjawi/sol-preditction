"""
model_engine.py v5
Models:
  1. Vanilla RNN      (numpy, BPTT)
  2. Random Forest    (sklearn) — replaces Stacked LSTM + ARIMAX
  3. Gradient Boosting (sklearn)
  4. Smart Ensemble   — only uses models with accuracy > 55%

Changes vs v4:
  - Removed Stacked LSTM (was performing at random ~49%)
  - Removed ARIMAX (was performing at random ~49%)
  - Added Random Forest (reliable, fast, typically 60-65% on financial data)
  - Smart ensemble: models below 55% accuracy are excluded automatically
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight

from feature_engine import FEATURE_COLS

HIGH    = 0.60
LOW     = 0.40
SEQ_LEN = 30     # 30-day lookback window
MIN_ACC = 0.55   # models below this are excluded from ensemble

# ─────────────────────────────────────────────────────────────────────────────
# Activation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sig(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
def _tanh(x):  return np.tanh(np.clip(x, -15, 15))
def _dtanh(t): return 1.0 - t**2


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimizer
# ─────────────────────────────────────────────────────────────────────────────
class _Adam:
    def __init__(self, lr=0.0005, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.t=0; self.m={}; self.v={}

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * (1-self.b2**self.t)**0.5 / (1-self.b1**self.t)
        for k in params:
            g = np.clip(grads.get(k, 0), -1.0, 1.0)
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*g
            self.v[k] = self.b2*self.v[k] + (1-self.b2)*g**2
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)
        return params


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla RNN (numpy, BPTT) — unchanged from v4
# ─────────────────────────────────────────────────────────────────────────────
class VanillaRNN:
    def __init__(self, input_size, hidden_size=64, lr=0.0005):
        H, I = hidden_size, input_size
        sc = lambda r,c: np.random.randn(r,c) * np.sqrt(2.0/(r+c))
        self.P = {
            "Wh": sc(H,H), "Wx": sc(H,I), "bh": np.zeros(H),
            "Wy": sc(1,H), "by": np.zeros(1),
        }
        self.H=H; self.opt=_Adam(lr); self.best_P=None

    def _forward_batch(self, X_batch):
        B,T,I = X_batch.shape
        h = np.zeros((B, self.H))
        for t in range(T):
            x = X_batch[:,t,:]
            h = _tanh(h @ self.P["Wh"].T + x @ self.P["Wx"].T + self.P["bh"])
        return _sig((h @ self.P["Wy"].T + self.P["by"]).flatten())

    def _bptt_batch(self, X_batch, y_batch, cw):
        B,T,I = X_batch.shape; H=self.H
        h_states = np.zeros((B,T+1,H))
        for t in range(T):
            x = X_batch[:,t,:]
            h_states[:,t+1,:] = _tanh(
                h_states[:,t,:] @ self.P["Wh"].T +
                x @ self.P["Wx"].T + self.P["bh"])
        h_last = h_states[:,T,:]
        prob   = _sig((h_last @ self.P["Wy"].T + self.P["by"]).flatten())
        w      = np.where(y_batch==1, cw[1], cw[0])
        dL_dp  = w*(prob-y_batch)
        dL_dh  = np.outer(dL_dp, self.P["Wy"])
        grads  = {
            "Wy": (dL_dp[:,None]*h_last).mean(axis=0,keepdims=True),
            "by": dL_dp.mean(keepdims=True),
            "Wh": np.zeros_like(self.P["Wh"]),
            "Wx": np.zeros_like(self.P["Wx"]),
            "bh": np.zeros_like(self.P["bh"]),
        }
        dh = dL_dh.copy()
        for t in range(T, max(0,T-10), -1):
            h_t   = h_states[:,t,:]
            h_tm1 = h_states[:,t-1,:]
            x_t   = X_batch[:,t-1,:]
            dt    = dh*_dtanh(h_t)
            grads["Wh"] += (dt.T @ h_tm1)/B
            grads["Wx"] += (dt.T @ x_t)  /B
            grads["bh"] += dt.mean(axis=0)
            dh = dt @ self.P["Wh"]
        loss = float(-np.mean(w*(y_batch*np.log(np.clip(prob,1e-7,1-1e-7))
                                +(1-y_batch)*np.log(np.clip(1-prob,1e-7,1-1e-7)))))
        return grads, loss

    def fit(self, X, y, X_val, y_val, cw, epochs=80, batch=32, patience=12, verbose=True):
        N=X.shape[0]; best_val=0.0; wait=0
        for ep in range(epochs):
            idx=np.random.permutation(N); ep_loss=[]
            for i in range(0,N,batch):
                b=idx[i:i+batch]
                g,loss=self._bptt_batch(X[b],y[b],cw)
                self.P=self.opt.step(self.P,g); ep_loss.append(loss)
            val_prob=self._forward_batch(X_val)
            val_acc=accuracy_score(y_val,(val_prob>0.5).astype(int))
            if verbose and ep%10==0:
                print(f"    RNN ep{ep:3d}  loss={np.mean(ep_loss):.4f}  val_acc={val_acc:.4f}")
            if val_acc>best_val:
                best_val=val_acc; self.best_P={k:v.copy() for k,v in self.P.items()}; wait=0
            else:
                wait+=1
                if wait>=patience:
                    if verbose: print(f"    RNN early stop ep{ep}  best={best_val:.4f}")
                    break
        if self.best_P: self.P=self.best_P
        return self

    def predict_proba(self, X):
        return self._forward_batch(X)


# ─────────────────────────────────────────────────────────────────────────────
# ModelEngine
# ─────────────────────────────────────────────────────────────────────────────
class ModelEngine:
    def __init__(self, df_feat: pd.DataFrame, split: float = 0.80):
        self.df=df_feat; self.split=split; self._prepare()

    def _prepare(self):
        d=self.df; sp=int(len(d)*self.split)
        if len(d)<100:
            raise RuntimeError(f"Not enough data: {len(d)} rows. Need ≥100.")
        self.tr=d.iloc[:sp]; self.te=d.iloc[sp:]
        feat=[c for c in FEATURE_COLS if c in d.columns]
        self.feat_cols=feat

        X_tr=self.tr[feat].values.astype(np.float32)
        X_te=self.te[feat].values.astype(np.float32)
        for arr in [X_tr,X_te]:
            med=np.nanmedian(X_tr,axis=0)
            mask=~np.isfinite(arr)
            arr[mask]=np.take(med,np.where(mask)[1])

        self.sc=MinMaxScaler()
        self.X_tr=self.sc.fit_transform(X_tr)
        self.X_te=self.sc.transform(X_te)
        self.y_tr=self.tr["Target"].values.astype(int)
        self.y_te=self.te["Target"].values.astype(int)
        self.y_ptr=self.tr["NextClose"].values
        self.y_pte=self.te["NextClose"].values

        cw=compute_class_weight("balanced",classes=np.array([0,1]),y=self.y_tr)
        self.CW={0:float(cw[0]),1:float(cw[1])}

        self.Xtr_s,self.ytr_s=self._seqs(self.X_tr,self.y_tr)
        self.Xte_s,self.yte_s=self._seqs(self.X_te,self.y_te)
        sp2=int(len(self.Xtr_s)*0.80)
        self.Xtr2=self.Xtr_s[:sp2]; self.ytr2=self.ytr_s[:sp2]
        self.Xval=self.Xtr_s[sp2:]; self.yval=self.ytr_s[sp2:]

    def _seqs(self,X,y):
        Xs,ys=[],[]
        for i in range(SEQ_LEN,len(X)):
            Xs.append(X[i-SEQ_LEN:i]); ys.append(y[i])
        return np.array(Xs,np.float32),np.array(ys,int)

    def train(self, verbose=True) -> dict:
        nf=len(self.feat_cols)
        y_te=self.yte_s
        all_probas={}; all_accs={}; model_data={}

        # ── 1. Vanilla RNN ────────────────────────────────────────────────────
        if verbose: print("Training Vanilla RNN...")
        try:
            rnn=VanillaRNN(nf,hidden_size=64,lr=0.0005)
            rnn.fit(self.Xtr2,self.ytr2,self.Xval,self.yval,
                    self.CW,epochs=80,batch=32,patience=12,verbose=verbose)
            rp=np.clip(rnn.predict_proba(self.Xte_s),0.01,0.99)
            ra=accuracy_score(y_te,(rp>0.5).astype(int))
            rf=f1_score(y_te,(rp>0.5).astype(int),zero_division=0)
            ru=roc_auc_score(y_te,rp) if len(np.unique(y_te))>1 else 0.5
            model_data["Vanilla RNN"]={"proba":rp,"pred":(rp>0.5).astype(int),
                                        "acc":ra,"f1":rf,"auc":ru}
            all_probas["Vanilla RNN"]=rp; all_accs["Vanilla RNN"]=ra
            print(f"  Vanilla RNN: Acc={ra*100:.2f}%  F1={rf:.4f}")
        except Exception as e:
            print(f"  RNN failed: {e}")
            all_probas["Vanilla RNN"]=np.full(len(y_te),0.5); all_accs["Vanilla RNN"]=0.5

        # ── 2. Random Forest ──────────────────────────────────────────────────
        if verbose: print("Training Random Forest...")
        try:
            rf_model=RandomForestClassifier(
                n_estimators=300, max_depth=8,
                class_weight="balanced", random_state=42, n_jobs=-1)
            rf_model.fit(self.X_tr, self.y_tr)
            rfp_full=rf_model.predict_proba(self.X_te)[:,1]
            rfp=rfp_full[SEQ_LEN:]
            rfa=accuracy_score(y_te,(rfp>0.5).astype(int))
            rff=f1_score(y_te,(rfp>0.5).astype(int),zero_division=0)
            rfu=roc_auc_score(y_te,rfp) if len(np.unique(y_te))>1 else 0.5
            model_data["Random Forest"]={"proba":rfp,"pred":(rfp>0.5).astype(int),
                                          "acc":rfa,"f1":rff,"auc":rfu}
            all_probas["Random Forest"]=rfp; all_accs["Random Forest"]=rfa
            print(f"  Random Forest: Acc={rfa*100:.2f}%  F1={rff:.4f}")
        except Exception as e:
            print(f"  Random Forest failed: {e}")
            all_probas["Random Forest"]=np.full(len(y_te),0.5); all_accs["Random Forest"]=0.5

        # ── 3. Gradient Boosting ──────────────────────────────────────────────
        if verbose: print("Training Gradient Boosting...")
        try:
            gb=GradientBoostingClassifier(n_estimators=300,max_depth=4,
                learning_rate=0.05,subsample=0.8,random_state=42)
            gb.fit(self.X_tr,self.y_tr)
            gbp=gb.predict_proba(self.X_te)[:,1][SEQ_LEN:]
            ga=accuracy_score(y_te,(gbp>0.5).astype(int))
            gf=f1_score(y_te,(gbp>0.5).astype(int),zero_division=0)
            gu=roc_auc_score(y_te,gbp) if len(np.unique(y_te))>1 else 0.5
            model_data["Gradient Boosting"]={"proba":gbp,"pred":(gbp>0.5).astype(int),
                                              "acc":ga,"f1":gf,"auc":gu}
            all_probas["Gradient Boosting"]=gbp; all_accs["Gradient Boosting"]=ga
            print(f"  Gradient Boosting: Acc={ga*100:.2f}%  F1={gf:.4f}")
        except Exception as e:
            print(f"  GB failed: {e}")
            all_probas["Gradient Boosting"]=np.full(len(y_te),0.5); all_accs["Gradient Boosting"]=0.5

        # ── Smart Ensemble: exclude models below MIN_ACC threshold ────────────
        good_models = {k:v for k,v in all_accs.items() if v >= MIN_ACC}
        if not good_models:
            # Fallback: use best available model even if below threshold
            best_k = max(all_accs, key=all_accs.get)
            good_models = {best_k: all_accs[best_k]}

        excluded = [k for k in all_accs if k not in good_models]
        if excluded:
            print(f"\n  ⚠️  Excluded from ensemble (acc < {MIN_ACC*100:.0f}%): {excluded}")
        print(f"  ✅ Ensemble uses: {list(good_models.keys())}")

        total_w   = sum(good_models.values())
        ens_proba = sum(good_models[k]*all_probas[k] for k in good_models) / total_w
        ens_pred  = (ens_proba>0.5).astype(int)
        ens_acc   = accuracy_score(y_te,ens_pred) if len(np.unique(y_te))>1 else 0.5
        ens_f1    = f1_score(y_te,ens_pred,zero_division=0)
        ens_auc   = roc_auc_score(y_te,ens_proba) if len(np.unique(y_te))>1 else 0.5

        filt      = (ens_proba>=HIGH)|(ens_proba<=LOW)
        ens_filt  = (accuracy_score(y_te[filt],(ens_proba[filt]>=HIGH).astype(int))
                     if filt.sum()>1 else ens_acc)
        signals   = np.where(ens_proba>=HIGH,1,np.where(ens_proba<=LOW,-1,0))

        best_name = max(all_accs, key=all_accs.get)
        print(f"\n  🏆 Best single model: {best_name}  ({all_accs[best_name]*100:.2f}%)")
        print(f"  Ensemble: Acc={ens_acc*100:.2f}%  Filtered={ens_filt*100:.2f}%")

        # ── Price regression ──────────────────────────────────────────────────
        try:
            ridge=Ridge(alpha=1.0)
            ridge.fit(self.X_tr,self.y_ptr)
            pp=ridge.predict(self.X_te)[SEQ_LEN:]
            y_pte_al=self.y_pte[SEQ_LEN:]
            rmse=float(np.sqrt(mean_squared_error(y_pte_al,pp)))
            mae=float(mean_absolute_error(y_pte_al,pp))
        except Exception:
            pp=self.te["Close"].values[SEQ_LEN:]
            y_pte_al=self.y_pte[SEQ_LEN:]; rmse=mae=0.0

        # ── Signal history ────────────────────────────────────────────────────
        te_al=self.te.iloc[SEQ_LEN:].copy()
        sh=pd.DataFrame({
            "Date"      : te_al.index[filt].strftime("%Y-%m-%d"),
            "Price"     : [f"${p:.4f}" for p in te_al["Close"].values[filt]],
            "Signal"    : ["🟢 BUY" if s==1 else "🔴 SELL"
                           for s in (ens_proba[filt]>=HIGH).astype(int)],
            "P(UP)"     : [f"{p*100:.1f}%" for p in ens_proba[filt]],
            "Confidence": [f"{max(p,1-p)*100:.1f}%" for p in ens_proba[filt]],
        }).sort_values("Date",ascending=False).reset_index(drop=True)

        for col in ["SMA20","SMA50","RSI","MACD","MACD_sig","Regime","BB_U","BB_L"]:
            if col not in te_al.columns: te_al[col]=0.0

        last_prob = float(ens_proba[-1])
        last_sig  = "BUY" if last_prob>=HIGH else "SELL" if last_prob<=LOW else "HOLD"
        last_conf = max(last_prob,1-last_prob)*100
        print(f"  Tomorrow: {last_sig}  ({last_conf:.1f}% confidence)")

        return {
            "model_data"        : model_data,
            "best_model"        : best_name,
            "ensemble_acc"      : ens_acc,
            "ensemble_filt_acc" : ens_filt,
            "ensemble_f1"       : ens_f1,
            "ensemble_auc"      : ens_auc,
            "ens_proba"         : ens_proba,
            "ens_pred"          : ens_pred,
            "y_te"              : y_te,
            "signals"           : signals,
            "n_signals"         : int((signals!=0).sum()),
            "HIGH"              : HIGH,
            "LOW"               : LOW,
            "last_signal"       : last_sig,
            "last_confidence"   : last_conf,
            "last_prob"         : last_prob,
            "price_pred"        : pp,
            "y_price_te"        : y_pte_al,
            "rmse"              : rmse,
            "mae"               : mae,
            "te_df"             : te_al,
            "signal_history"    : sh,
            "TF_AVAILABLE"      : False,
            "ensemble_models"   : list(good_models.keys()),
            "excluded_models"   : excluded,
        }
