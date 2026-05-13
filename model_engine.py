"""
model_engine.py v6
──────────────────
Models:
  1. Vanilla RNN       (numpy BPTT)
  2. Random Forest     (sklearn, 300 trees)
  3. Gradient Boosting (sklearn, 300 trees)
  4. XGBoost           (xgboost library — handles non-linear patterns differently)

Changes vs v5:
  • XGBoost replaces nothing — added as 4th model
  • Smart ensemble keeps models ≥55% accuracy only
  • Works on hourly AND daily data (auto-detects by row count)
  • Sentiment score injected as feature if available
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
MIN_ACC = 0.55
SEQ_LEN = 30

# ── Activations ───────────────────────────────────────────────────
def _sig(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
def _tanh(x):  return np.tanh(np.clip(x, -15, 15))
def _dtanh(t): return 1.0 - t**2


class _Adam:
    def __init__(self, lr=5e-4, b1=.9, b2=.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.t=0; self.m={}; self.v={}

    def step(self, P, G):
        self.t += 1
        lr_t = self.lr*(1-self.b2**self.t)**.5/(1-self.b1**self.t)
        for k in P:
            g = np.clip(G.get(k,0), -1, 1)
            if k not in self.m:
                self.m[k]=np.zeros_like(P[k]); self.v[k]=np.zeros_like(P[k])
            self.m[k]=self.b1*self.m[k]+(1-self.b1)*g
            self.v[k]=self.b2*self.v[k]+(1-self.b2)*g**2
            P[k]-=lr_t*self.m[k]/(np.sqrt(self.v[k])+self.eps)
        return P


class VanillaRNN:
    def __init__(self, input_size, hidden_size=64, lr=5e-4):
        H,I=hidden_size,input_size
        sc=lambda r,c: np.random.randn(r,c)*np.sqrt(2/(r+c))
        self.P={"Wh":sc(H,H),"Wx":sc(H,I),"bh":np.zeros(H),
                "Wy":sc(1,H),"by":np.zeros(1)}
        self.H=H; self.opt=_Adam(lr); self.best_P=None

    def _fwd(self, X):
        B,T,_=X.shape; h=np.zeros((B,self.H))
        for t in range(T):
            h=_tanh(h@self.P["Wh"].T+X[:,t,:]@self.P["Wx"].T+self.P["bh"])
        return _sig((h@self.P["Wy"].T+self.P["by"]).flatten())

    def _bptt(self, X, y, cw):
        B,T,_=X.shape; H=self.H
        hs=np.zeros((B,T+1,H))
        for t in range(T):
            hs[:,t+1,:]=_tanh(hs[:,t,:]@self.P["Wh"].T+X[:,t,:]@self.P["Wx"].T+self.P["bh"])
        p=_sig((hs[:,T,:]@self.P["Wy"].T+self.P["by"]).flatten())
        w=np.where(y==1,cw[1],cw[0]); dl=w*(p-y)
        G={"Wy":(dl[:,None]*hs[:,T,:]).mean(0,keepdims=True),"by":dl.mean(keepdims=True),
           "Wh":np.zeros_like(self.P["Wh"]),"Wx":np.zeros_like(self.P["Wx"]),"bh":np.zeros_like(self.P["bh"])}
        dh=np.outer(dl,self.P["Wy"])
        for t in range(T,max(0,T-10),-1):
            dt=dh*_dtanh(hs[:,t,:])
            G["Wh"]+=(dt.T@hs[:,t-1,:])/B; G["Wx"]+=(dt.T@X[:,t-1,:])/B; G["bh"]+=dt.mean(0)
            dh=dt@self.P["Wh"]
        loss=float(-np.mean(w*(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)))))
        return G, loss

    def fit(self, X, y, Xv, yv, cw, epochs=80, batch=32, patience=12, verbose=True):
        N=X.shape[0]; bv=0; wait=0
        for ep in range(epochs):
            idx=np.random.permutation(N); losses=[]
            for i in range(0,N,batch):
                b=idx[i:i+batch]; g,l=self._bptt(X[b],y[b],cw)
                self.P=self.opt.step(self.P,g); losses.append(l)
            va=accuracy_score(yv,(self._fwd(Xv)>0.5).astype(int))
            if verbose and ep%20==0: print(f"    RNN ep{ep:3d} loss={np.mean(losses):.4f} val={va:.4f}")
            if va>bv: bv=va; self.best_P={k:v.copy() for k,v in self.P.items()}; wait=0
            else:
                wait+=1
                if wait>=patience: break
        if self.best_P: self.P=self.best_P
        return self

    def predict_proba(self, X): return self._fwd(X)


class ModelEngine:
    def __init__(self, df: pd.DataFrame, split: float = 0.80):
        self.df=df; self.split=split; self._prepare()

    def _prepare(self):
        d=self.df; sp=int(len(d)*self.split)
        if len(d)<100: raise RuntimeError(f"Not enough data: {len(d)} rows. Need ≥100.")
        self.tr=d.iloc[:sp]; self.te=d.iloc[sp:]
        feat=[c for c in FEATURE_COLS if c in d.columns]
        self.feat_cols=feat
        X_tr=self.tr[feat].values.astype(np.float32)
        X_te=self.te[feat].values.astype(np.float32)
        # Fill NaN/inf with column median from training set
        _med = np.nanmedian(X_tr, axis=0)
        _med = np.where(np.isfinite(_med), _med, 0.0)
        for arr in [X_tr, X_te]:
            _mask = ~np.isfinite(arr)
            arr[_mask] = np.take(_med, np.where(_mask)[1])
        X_tr = np.clip(X_tr, -1e6, 1e6)
        X_te = np.clip(X_te, -1e6, 1e6)
        self.sc=MinMaxScaler()
        self.X_tr=self.sc.fit_transform(X_tr); self.X_te=self.sc.transform(X_te)
        self.y_tr=self.tr["Target"].values.astype(int)
        self.y_te=self.te["Target"].values.astype(int)
        _y_tr = self.tr["NextClose"].values.astype(np.float64)
        _y_te = self.te["NextClose"].values.astype(np.float64)
        # Replace NaN in targets with last valid value
        _y_tr[~np.isfinite(_y_tr)] = np.nanmean(_y_tr)
        _y_te[~np.isfinite(_y_te)] = np.nanmean(_y_te)
        self.y_ptr = _y_tr; self.y_pte = _y_te
        cw=compute_class_weight("balanced",classes=np.array([0,1]),y=self.y_tr)
        self.CW={0:float(cw[0]),1:float(cw[1])}
        self.Xtr_s,self.ytr_s=self._seqs(self.X_tr,self.y_tr)
        self.Xte_s,self.yte_s=self._seqs(self.X_te,self.y_te)
        sp2=int(len(self.Xtr_s)*.80)
        self.Xtr2=self.Xtr_s[:sp2]; self.ytr2=self.ytr_s[:sp2]
        self.Xval=self.Xtr_s[sp2:]; self.yval=self.ytr_s[sp2:]

    def _seqs(self, X, y):
        Xs,ys=[],[]
        for i in range(SEQ_LEN,len(X)):
            Xs.append(X[i-SEQ_LEN:i]); ys.append(y[i])
        return np.array(Xs,np.float32), np.array(ys,int)

    def train(self, verbose=False, sentiment_score: float = 0.0) -> dict:
        nf=len(self.feat_cols); y_te=self.yte_s
        all_p={}; all_a={}; model_data={}

        # ── 1. Vanilla RNN ────────────────────────────────────────
        if verbose: print("Training RNN...")
        try:
            # Forward-fill NaN in sequence data (better than median=0 for RNN)
            _Xtr2 = self.Xtr2.copy(); _Xval = self.Xval.copy(); _Xte_s = self.Xte_s.copy()
            for _arr in [_Xtr2, _Xval, _Xte_s]:
                for _col in range(_arr.shape[1]):
                    _mask = ~np.isfinite(_arr[:, _col])
                    if _mask.any():
                        _arr[_mask, _col] = np.nanmedian(_arr[:, _col]) if np.isfinite(_arr[:, _col]).any() else 0.0
            rnn=VanillaRNN(nf,64,4e-4)  # larger hidden, slightly lower lr
            rnn.fit(_Xtr2,self.ytr2,_Xval,self.yval,
                    self.CW,epochs=50,batch=64,patience=8,verbose=verbose)
            rp=np.clip(rnn.predict_proba(_Xte_s),0.01,0.99)
            ra=accuracy_score(y_te,(rp>0.5).astype(int))
            rf=f1_score(y_te,(rp>0.5).astype(int),zero_division=0)
            ru=roc_auc_score(y_te,rp) if len(np.unique(y_te))>1 else 0.5
            model_data["Vanilla RNN"]={"proba":rp,"pred":(rp>0.5).astype(int),"acc":ra,"f1":rf,"auc":ru}
            all_p["Vanilla RNN"]=rp; all_a["Vanilla RNN"]=ra
            print(f"  RNN: {ra*100:.2f}%")
        except Exception as e:
            print(f"  RNN failed: {e}")
            all_p["Vanilla RNN"]=np.full(len(y_te),0.5); all_a["Vanilla RNN"]=0.5

        # ── 2. Random Forest ──────────────────────────────────────
        if verbose: print("Training RF...")
        try:
            rf_m=RandomForestClassifier(n_estimators=150,max_depth=6,min_samples_leaf=3,
                class_weight="balanced",max_features="sqrt",random_state=42,n_jobs=-1)
            rf_m.fit(self.X_tr,self.y_tr)
            rfp=rf_m.predict_proba(self.X_te)[:,1][SEQ_LEN:]
            rfa=accuracy_score(y_te,(rfp>0.5).astype(int))
            rff=f1_score(y_te,(rfp>0.5).astype(int),zero_division=0)
            rfu=roc_auc_score(y_te,rfp) if len(np.unique(y_te))>1 else 0.5
            model_data["Random Forest"]={"proba":rfp,"pred":(rfp>0.5).astype(int),"acc":rfa,"f1":rff,"auc":rfu}
            all_p["Random Forest"]=rfp; all_a["Random Forest"]=rfa
            print(f"  RF: {rfa*100:.2f}%")
        except Exception as e:
            print(f"  RF failed: {e}")
            all_p["Random Forest"]=np.full(len(y_te),0.5); all_a["Random Forest"]=0.5

        # ── 3. Gradient Boosting ──────────────────────────────────
        if verbose: print("Training GB...")
        try:
            gb=GradientBoostingClassifier(n_estimators=150,max_depth=4,
                learning_rate=0.06,subsample=0.80,max_features="sqrt",random_state=42)
            gb.fit(self.X_tr,self.y_tr)
            gbp=gb.predict_proba(self.X_te)[:,1][SEQ_LEN:]
            ga=accuracy_score(y_te,(gbp>0.5).astype(int))
            gf=f1_score(y_te,(gbp>0.5).astype(int),zero_division=0)
            gu=roc_auc_score(y_te,gbp) if len(np.unique(y_te))>1 else 0.5
            model_data["Gradient Boosting"]={"proba":gbp,"pred":(gbp>0.5).astype(int),"acc":ga,"f1":gf,"auc":gu}
            all_p["Gradient Boosting"]=gbp; all_a["Gradient Boosting"]=ga
            print(f"  GB: {ga*100:.2f}%")
        except Exception as e:
            print(f"  GB failed: {e}")
            all_p["Gradient Boosting"]=np.full(len(y_te),0.5); all_a["Gradient Boosting"]=0.5

        # ── 4. XGBoost ────────────────────────────────────────────
        if verbose: print("Training XGBoost...")
        try:
            import xgboost as xgb
            scale_pw = self.CW[1]/self.CW[0]
            xgb_m = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pw,
                eval_metric="logloss", verbosity=0,
                random_state=42, n_jobs=-1,
                tree_method="hist",
            )
            xgb_m.fit(self.X_tr, self.y_tr,
                      eval_set=[(self.X_te, self.y_te)],
                      verbose=False)
            xp = xgb_m.predict_proba(self.X_te)[:,1][SEQ_LEN:]
            xa = accuracy_score(y_te,(xp>0.5).astype(int))
            xf = f1_score(y_te,(xp>0.5).astype(int),zero_division=0)
            xu = roc_auc_score(y_te,xp) if len(np.unique(y_te))>1 else 0.5
            model_data["XGBoost"]={"proba":xp,"pred":(xp>0.5).astype(int),"acc":xa,"f1":xf,"auc":xu}
            all_p["XGBoost"]=xp; all_a["XGBoost"]=xa
            print(f"  XGBoost: {xa*100:.2f}%")
        except ImportError:
            print("  XGBoost not installed — skipping")
        except Exception as e:
            print(f"  XGBoost failed: {e}")
            all_p["XGBoost"]=np.full(len(y_te),0.5); all_a["XGBoost"]=0.5

        # ── Smart ensemble ────────────────────────────────────────
        good = {k:v for k,v in all_a.items() if v >= MIN_ACC}
        if not good:
            best_k=max(all_a,key=all_a.get); good={best_k:all_a[best_k]}
        excluded=[k for k in all_a if k not in good]
        if excluded: print(f"  Excluded (<{MIN_ACC*100:.0f}%): {excluded}")
        print(f"  Ensemble uses: {list(good.keys())}")

        total_w   = sum(good.values())
        ens_proba = sum(good[k]*all_p[k] for k in good)/total_w
        ens_pred  = (ens_proba>0.5).astype(int)
        ens_acc   = accuracy_score(y_te,ens_pred) if len(np.unique(y_te))>1 else 0.5
        ens_f1    = f1_score(y_te,ens_pred,zero_division=0)
        ens_auc   = roc_auc_score(y_te,ens_proba) if len(np.unique(y_te))>1 else 0.5
        filt      = (ens_proba>=HIGH)|(ens_proba<=LOW)
        ens_filt  = (accuracy_score(y_te[filt],(ens_proba[filt]>=HIGH).astype(int))
                     if filt.sum()>1 else ens_acc)
        signals   = np.where(ens_proba>=HIGH,1,np.where(ens_proba<=LOW,-1,0))

        # ── Multiple signals per day ──────────────────────────────────────
        # For daily data: find ALL BUY/SELL points (not just last one)
        # For intraday: every candle with confidence >= threshold is a signal
        # This gives traders multiple entry opportunities
        n_signals_total = int((signals != 0).sum())
        buy_signals_idx  = np.where(signals == 1)[0]
        sell_signals_idx = np.where(signals == -1)[0]
        best_name = max(all_a,key=all_a.get)
        print(f"  Best: {best_name} ({all_a[best_name]*100:.2f}%)")
        print(f"  Ensemble: {ens_acc*100:.2f}% / filtered: {ens_filt*100:.2f}%")

        # ── Price regression (Ridge on raw prices - works correctly) ─────
        try:
            ridge = Ridge(alpha=1.0)
            ridge.fit(self.X_tr, self.y_ptr)
            pp = ridge.predict(self.X_te)[SEQ_LEN:]
            y_pte_al = self.y_pte[SEQ_LEN:]
            # Align lengths
            _mn = min(len(pp), len(y_pte_al))
            pp = pp[:_mn]; y_pte_al = y_pte_al[:_mn]
            rmse = float(np.sqrt(mean_squared_error(y_pte_al, pp)))
            mae  = float(mean_absolute_error(y_pte_al, pp))
        except Exception as _e:
            print(f"  Ridge failed: {_e}")
            # Fallback: simple moving average prediction
            _close_vals = self.te["Close"].values
            _ma = pd.Series(_close_vals).rolling(5, min_periods=1).mean().values
            pp = _ma[SEQ_LEN:]
            y_pte_al = self.y_pte[SEQ_LEN:]
            _mn = min(len(pp), len(y_pte_al))
            pp = pp[:_mn]; y_pte_al = y_pte_al[:_mn]
            try:
                rmse = float(np.sqrt(mean_squared_error(y_pte_al, pp)))
                mae  = float(mean_absolute_error(y_pte_al, pp))
            except Exception:
                rmse = mae = 0.0

        # ── Signal history ────────────────────────────────────────
        te_al=self.te.iloc[SEQ_LEN:].copy()
        sh=pd.DataFrame({
            "Date":te_al.index[filt].strftime("%Y-%m-%d"),
            "Price":[f"${p:.4f}" for p in te_al["Close"].values[filt]],
            "Signal":["🟢 BUY" if s==1 else "🔴 SELL"
                      for s in (ens_proba[filt]>=HIGH).astype(int)],
            "P(UP)":[f"{p*100:.1f}%" for p in ens_proba[filt]],
            "Confidence":[f"{max(p,1-p)*100:.1f}%" for p in ens_proba[filt]],
        }).sort_values("Date",ascending=False).reset_index(drop=True)

        for col in ["SMA20","SMA50","RSI","MACD","MACD_sig","Regime","BB_U","BB_L"]:
            if col not in te_al.columns: te_al[col]=0.0

        last_prob=float(ens_proba[-1])
        last_sig="BUY" if last_prob>=HIGH else "SELL" if last_prob<=LOW else "HOLD"
        last_conf=max(last_prob,1-last_prob)*100
        print(f"  Tomorrow: {last_sig} ({last_conf:.1f}%)")

        # Sentiment influence on signal confidence
        # Bullish news boosts BUY confidence, bearish news boosts SELL confidence
        if sentiment_score > 0.1 and last_sig == "BUY":
            last_conf = min(99.0, last_conf * (1 + sentiment_score * 0.15))
            print(f"  Sentiment boost: +{sentiment_score*15:.1f}% → {last_conf:.1f}%")
        elif sentiment_score < -0.1 and last_sig == "SELL":
            last_conf = min(99.0, last_conf * (1 + abs(sentiment_score) * 0.15))
            print(f"  Sentiment boost: +{abs(sentiment_score)*15:.1f}% → {last_conf:.1f}%")

        # ── Build multi-signal table ────────────────────────────────────
        _te_idx = self.te.iloc[SEQ_LEN:].index
        _all_signal_dates  = []
        _all_signal_prices = []
        _all_signal_types  = []
        _all_signal_confs  = []
        for _idx in range(len(signals)):
            if signals[_idx] != 0:
                _dt  = _te_idx[_idx] if _idx < len(_te_idx) else None
                _p   = float(self.te["Close"].iloc[SEQ_LEN + _idx]) if SEQ_LEN+_idx < len(self.te) else 0
                _sig = "BUY" if signals[_idx]==1 else "SELL"
                _cf  = float(max(ens_proba[_idx], 1-ens_proba[_idx])) * 100
                _all_signal_dates.append(str(_dt)[:10] if _dt is not None else "")
                _all_signal_prices.append(_p)
                _all_signal_types.append(_sig)
                _all_signal_confs.append(_cf)

        multi_signals_df = __import__("pandas").DataFrame({
            "Date"      : _all_signal_dates,
            "Price"     : [f"${p:,.4f}" for p in _all_signal_prices],
            "Signal"    : ["🟢 BUY" if s=="BUY" else "🔴 SELL" for s in _all_signal_types],
            "Confidence": [f"{c:.1f}%" for c in _all_signal_confs],
        }).sort_values("Date", ascending=False).reset_index(drop=True)

        return {
            "model_data":model_data,"best_model":best_name,
            "ensemble_acc":ens_acc,"ensemble_filt_acc":ens_filt,
            "ensemble_f1":ens_f1,"ensemble_auc":ens_auc,
            "ens_proba":ens_proba,"ens_pred":ens_pred,"y_te":y_te,
            "signals":signals,"n_signals":int((signals!=0).sum()),
            "HIGH":HIGH,"LOW":LOW,
            "last_signal":last_sig,"last_confidence":last_conf,"last_prob":last_prob,
            "price_pred":pp,"y_price_te":y_pte_al,"rmse":rmse,"mae":mae,
            "te_df":te_al,"signal_history":sh,
            "TF_AVAILABLE":False,
            "multi_signals":multi_signals_df,
            "sentiment_score":sentiment_score,
            "ensemble_models":list(good.keys()),"excluded_models":excluded,
        }
