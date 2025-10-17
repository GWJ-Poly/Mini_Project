
import os
import gc
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# sklearn / saving
import joblib

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------- User config -----------------
DATA_DIR = './data'
SALES_FILE = os.path.join(DATA_DIR, 'sales_train_validation.csv')
CALENDAR_FILE = os.path.join(DATA_DIR, 'calendar.csv')
PRICES_FILE = os.path.join(DATA_DIR, 'sell_prices.csv')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

OUTPUT_SALES_LONG = 'sales_long.feather'
OUTPUT_SAMPLES = 'samples.feather'
OUTPUT_SAMPLES_HIST = 'samples_hist.feather'
OUTPUT_SUB = 'submission_lstm_adapter.csv'
MODEL_SAVE = 'best_lstm_adapter_model.pt'
META_SAVE = 'lstm_adapter_meta.npz'

VALIDATION_PRED_CSV = 'validation_preds_best.csv'
FINAL_PRED_CSV = 'submission_preds_final.csv'

USE_GPU = True
GPU_DEVICE_ID = 0

# training hyperparams
SELECTED_MODEL = 'lstm_adapter'  # 'lstm_adapter' runs this pipeline
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# model hyperparams
HIST_LEN = 28
LSTM_HIDDEN = 256
LSTM_LAYERS = 3
OUT_HORIZON = 28
ADAPTER_FILMDIM = 128
ADAPTER_ATTN_HEADS = 3
CONTINUOUS_FEATURES = ['roll_mean_7', 'sell_price', 'price_lag_7']
CATEGORICAL_FEATURES = ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id', 'wday', 'month']

USE_LOG = False   # optional: train with MSE on log1p space (recommended for long-tail)

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- Data utilities (same FE as your script) ----------------
def load_data():
    sales = pd.read_csv(SALES_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)
    prices = pd.read_csv(PRICES_FILE)
    return sales, calendar, prices

def sales_wide_to_long(sales_wide):
    id_cols = [c for c in sales_wide.columns if c.startswith('id') or c in ['item_id','dept_id','cat_id','store_id','state_id']]
    date_cols = [c for c in sales_wide.columns if c.startswith('d_')]
    sales_long = sales_wide.melt(id_vars=id_cols, value_vars=date_cols, var_name='d', value_name='sales')
    return sales_long

def map_calendar(sales_long, calendar):
    cal = calendar[['d','date','wm_yr_wk','weekday','wday','month','year',
                    'snap_CA','snap_TX','snap_WI',
                    'event_name_1','event_type_1','event_name_2','event_type_2']].copy()
    cal['date'] = pd.to_datetime(cal['date'])
    sales_long = sales_long.merge(cal, on='d', how='left')
    sales_long.rename(columns={'date':'date_dt'}, inplace=True)
    sales_long['date'] = pd.to_datetime(sales_long['date_dt'])
    sales_long.drop(columns=['date_dt'], inplace=True)
    return sales_long

def merge_prices(sales_long, prices, calendar):
    if 'wm_yr_wk' not in sales_long.columns:
        sales_long = sales_long.merge(calendar[['d','wm_yr_wk']], on='d', how='left')
    prices_sub = prices[['wm_yr_wk','store_id','item_id','sell_price']]
    sales_long = sales_long.merge(prices_sub, on=['wm_yr_wk','store_id','item_id'], how='left')
    return sales_long

def basic_clean(df):
    df['sales'] = df['sales'].clip(lower=0)
    if 'sell_price' in df.columns:
        df['sell_price'] = df['sell_price'].ffill().bfill()
    df['date'] = pd.to_datetime(df['date'])
    return df

def make_lags_rolls_numpy(df, lags=[1,7,28], rolls=[7,28]):
    df_sorted = df.sort_values(['id','date']).copy()
    ids = df_sorted['id'].unique()
    for lag in lags:
        df_sorted[f'lag_{lag}'] = np.nan
    for r in rolls:
        df_sorted[f'roll_mean_{r}'] = np.nan
        df_sorted[f'roll_std_{r}'] = np.nan
    df_sorted['price_lag_7'] = np.nan
    df_sorted['is_weekend'] = df_sorted['wday'].isin([0,6]).astype(int)
    group_iter = df_sorted.groupby('id')
    total = len(ids)
    for _id, df_item in tqdm(group_iter, total=total, desc='make_lags_rolls'):
        idx = df_item.index
        sales_arr = df_item['sales'].values.astype(float)
        price_arr = df_item['sell_price'].values.astype(float)
        for lag in lags:
            arr = np.roll(sales_arr, lag)
            if lag > 0:
                arr[:lag] = np.nan
            df_sorted.loc[idx, f'lag_{lag}'] = arr
        shifted = pd.Series(sales_arr).shift(1)
        for r in rolls:
            df_sorted.loc[idx, f'roll_mean_{r}'] = shifted.rolling(r).mean().values
            df_sorted.loc[idx, f'roll_std_{r}']  = shifted.rolling(r).std().values
        arr_price = np.roll(price_arr, 7)
        arr_price[:7] = np.nan
        df_sorted.loc[idx, 'price_lag_7'] = arr_price
    return df_sorted

def generate_samples(df, cutoffs, horizon=28, features=None):
    out_rows = []
    df_indexed = df.set_index(['id','date'])
    id_list = df['id'].unique()
    feats = features if features is not None else [
        'lag_1','lag_7','lag_28','roll_mean_7','roll_mean_28','roll_std_7',
        'sell_price','price_lag_7','wday','month','is_weekend'
    ]
    for cutoff in tqdm(cutoffs, desc='generate_samples'):
        cur_date = pd.to_datetime(cutoff)
        next_dates = [cur_date + timedelta(days=i) for i in range(1, horizon+1)]
        for _id in id_list:
            try:
                base = df_indexed.loc[(_id, cur_date)]
            except KeyError:
                continue
            row = {'id': _id, 'cutoff': cur_date}
            for f in feats:
                row[f] = base.get(f, np.nan)
            ok = True
            for h, nd in enumerate(next_dates, start=1):
                try:
                    row[f'out_{h}'] = df_indexed.loc[(_id, nd)]['sales']
                except Exception:
                    ok = False
                    break
            if ok:
                out_rows.append(row)
    df_samples = pd.DataFrame(out_rows)
    return df_samples

def generate_samples_with_history(df, cutoffs, horizon=28, history=28, features=None):
    out_rows = []
    df_indexed = df.set_index(['id','date'])
    id_list = df['id'].unique()
    feats = features if features is not None else [
        'lag_1','lag_7','lag_28','roll_mean_7','roll_mean_28','roll_std_7',
        'sell_price','price_lag_7','wday','month','is_weekend'
    ]
    for cutoff in tqdm(cutoffs, desc='generate_samples_hist'):
        cur_date = pd.to_datetime(cutoff)
        next_dates = [cur_date + timedelta(days=i) for i in range(1, horizon+1)]
        hist_dates = [cur_date - timedelta(days=i) for i in range(history, 0, -1)]
        for _id in id_list:
            try:
                base = df_indexed.loc[(_id, cur_date)]
            except KeyError:
                continue
            row = {'id': _id, 'cutoff': cur_date}
            for f in feats:
                row[f] = base.get(f, np.nan)
            ok = True
            for i, hd in enumerate(hist_dates, start=1):
                try:
                    row[f'hist_{i}'] = df_indexed.loc[(_id, hd)]['sales']
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            for h, nd in enumerate(next_dates, start=1):
                try:
                    row[f'out_{h}'] = df_indexed.loc[(_id, nd)]['sales']
                except Exception:
                    ok = False
                    break
            if ok:
                out_rows.append(row)
    df_samples = pd.DataFrame(out_rows)
    return df_samples

# ---------------- categorical mapping helpers ----------------
def build_cat_maps(df, cat_feats):
    maps = {}
    for c in cat_feats:
        if c in df.columns:
            vals = df[c].astype(str).fillna('-1')
            uniques = vals.unique().tolist()
            m = {v: i+1 for i, v in enumerate(uniques)}  # start from 1; 0 reserved for unk
            maps[c] = m
        else:
            maps[c] = {}
    return maps

def apply_map_series(s, cmap):
    return s.astype(str).fillna('-1').map(lambda x: cmap.get(x, 0)).astype(np.int64)

# ---------------- Updated build_latest_hist_rows (adds cont/cat at cutoff) ----------------
def build_latest_hist_rows(sales_long_df, cutoff_date, history=28, cat_feats=None, cont_feats=None):

    cat_feats = cat_feats or []
    cont_feats = cont_feats or []
    df_indexed = sales_long_df.set_index(['id','date'])
    ids = sales_long_df['id'].unique()
    rows = []
    hdates = [pd.to_datetime(cutoff_date) - timedelta(days=i) for i in range(history,0,-1)]
    for _id in ids:
        ok = True
        row = {'id': _id}
        # history
        for i, hd in enumerate(hdates, start=1):
            try:
                row[f'hist_{i}'] = df_indexed.loc[(_id, hd)]['sales']
            except Exception:
                ok = False
                break
        if not ok:
            continue
        # now add cont and cat features using the base (cutoff) row if possible
        try:
            base = df_indexed.loc[(_id, pd.to_datetime(cutoff_date))]
        except Exception:
            # cutoff row missing for this id -> skip
            continue
        for cf in cont_feats:
            # base is a pandas Series; use get
            row[cf] = base.get(cf, np.nan)
        for kf in cat_feats:
            row[kf] = base.get(kf, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------- Updated CombinedDataset & collate ----------------
class CombinedDataset(Dataset):
    def __init__(self, df, hist_len=28, cat_feats=None, cont_feats=None, cat_maps=None):

        self.df = df.reset_index(drop=True)
        self.hist_len = hist_len
        self.cat_feats = cat_feats or []
        self.cont_feats = cont_feats or []
        self.cat_maps = cat_maps or {}
        self.hist_cols = [f'hist_{i}' for i in range(1, hist_len+1)]

        # X_hist: ensure present
        if not set(self.hist_cols).issubset(set(self.df.columns)):
            raise RuntimeError(f"Dataset expected hist cols {self.hist_cols} in dataframe; missing some cols.")
        self.X_hist = self.df[self.hist_cols].values.astype(np.float32)  # (N,T)

        # cat arrays: if column missing, fallback to zeros
        self.cat_arrays = {}
        n = len(self.df)
        for c in self.cat_feats:
            if c in self.df.columns:
                cmap = self.cat_maps.get(c, {})
                self.cat_arrays[c] = apply_map_series(self.df[c], cmap).astype(np.int64).values
            else:
                self.cat_arrays[c] = np.zeros(n, dtype=np.int64)

        # cont array: if columns missing, build zeros with shape (n, 0) or fill missing columns
        cont_present = [c for c in self.cont_feats if c in self.df.columns]
        if len(cont_present) > 0:
            cont_arr_present = self.df[cont_present].fillna(0).values.astype(np.float32)
            if len(cont_present) != len(self.cont_feats):
                # expand to full order
                full = np.zeros((n, len(self.cont_feats)), dtype=np.float32)
                for i, c in enumerate(self.cont_feats):
                    if c in cont_present:
                        idx = cont_present.index(c)
                        full[:, i] = cont_arr_present[:, idx]
                    else:
                        full[:, i] = 0.0
                self.cont_array = full
            else:
                self.cont_array = cont_arr_present
        else:
            self.cont_array = np.zeros((n, 0), dtype=np.float32)

        # targets: if exist in df, else zeros (useful for inference)
        out_cols = [f'out_{h}' for h in range(1,29)]
        if set(out_cols).issubset(set(self.df.columns)):
            self.y = self.df[out_cols].values.astype(np.float32)
        else:
            self.y = np.zeros((n, len(out_cols)), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hist = self.X_hist[idx].astype(np.float32)           # numpy array
        cat_inputs = {c: int(self.cat_arrays[c][idx]) for c in self.cat_feats}
        cont = self.cont_array[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        return hist, cat_inputs, cont, y

def combined_collate(batch):
    """
    batch: list of tuples (hist_np, cat_dict, cont_np, y_np)
    returns: (hist_tensor(B,T), cat_tensor_dict, cont_tensor(B,cont_dim) or zeros (B,0), y_tensor(B,H))
    """
    hists = np.stack([b[0] for b in batch], axis=0)          # shape (B, T)
    cat_keys = list(batch[0][1].keys())
    cat_batch = {}
    for k in cat_keys:
        cat_batch[k] = torch.tensor([b[1][k] for b in batch], dtype=torch.long)

    cont0 = batch[0][2]
    if cont0 is None:
        cont_tensor = torch.zeros((len(batch), 0), dtype=torch.float32)
    else:
        cont_dim = cont0.shape[0]
        if cont_dim > 0:
            cont_arr = np.stack([b[2] for b in batch], axis=0)   # (B, cont_dim)
            cont_tensor = torch.tensor(cont_arr, dtype=torch.float32)
        else:
            cont_tensor = torch.zeros((len(batch), 0), dtype=torch.float32)

    ys = np.stack([b[3] for b in batch], axis=0)
    y_tensor = torch.tensor(ys, dtype=torch.float32)
    hist_tensor = torch.tensor(hists, dtype=torch.float32)
    return hist_tensor, cat_batch, cont_tensor, y_tensor

# ---------------- The original user LSTMNet (provided) ----------------
class LSTMNet(nn.Module):
    def __init__(self, seq_len, hidden_size=64, num_layers=3, out_dim=28):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, out_dim)
        )
    def forward(self, x):  # x: (batch, seq_len, 1)
        out, (hn, cn) = self.lstm(x)
        last = hn[-1]  # (batch, hidden)
        return self.fc(last)

# ---------------- Adapter components (simple, light) ----------------
class SmallFeatureEncoder(nn.Module):
    def __init__(self, cat_cardinalities=None, cont_dim=0, emb_dim=16, out_dim=64):
        super().__init__()
        cat_cardinalities = cat_cardinalities or {}
        self.embs = nn.ModuleDict()
        for k, v in cat_cardinalities.items():
            self.embs[k] = nn.Embedding(max(1, v), min(emb_dim, max(2, int(np.sqrt(v) + 1))))
        total_emb = sum(e.embedding_dim for e in self.embs.values())
        in_dim = total_emb + cont_dim
        if in_dim == 0:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
    def forward(self, cat_inputs: dict, cont_inputs=None):
        embs = []
        for k, emb in self.embs.items():
            embs.append(emb(cat_inputs[k]))
        if embs:
            x = torch.cat(embs, dim=-1)
        else:
            batch = cont_inputs.shape[0] if (cont_inputs is not None and cont_inputs.numel()>0) else 1
            device = cont_inputs.device if (cont_inputs is not None and cont_inputs.numel()>0) else torch.device('cpu')
            x = torch.zeros((batch, 0), device=device)
        if cont_inputs is not None and cont_inputs.numel()>0:
            x = torch.cat([x, cont_inputs], dim=-1)
        return self.mlp(x) if not isinstance(self.mlp, nn.Identity) else x

class M5SimpleAdapter(nn.Module):
    def __init__(self, hidden_size, hist_len=28, out_horizon=28,
                 cat_cardinalities=None, cont_dim=0, film_dim=64, attn_heads=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.hist_len = hist_len
        self.out_horizon = out_horizon

        self.hist_proj = nn.Linear(1, hidden_size)
        self.num_heads = attn_heads if hidden_size % attn_heads == 0 else 1
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=self.num_heads, batch_first=True)
        self.attn_ln = nn.LayerNorm(hidden_size)

        self.feat_enc = SmallFeatureEncoder(cat_cardinalities=cat_cardinalities, cont_dim=cont_dim, emb_dim=16, out_dim=film_dim)
        self.gamma = nn.Linear(film_dim, hidden_size)
        self.beta  = nn.Linear(film_dim, hidden_size)

        self.fuse_ln = nn.LayerNorm(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, out_horizon)
        )

    def forward(self, last_hidden, hist_seq=None, cat_inputs=None, cont_inputs=None, price_seq=None):
        B = last_hidden.shape[0]
        device = last_hidden.device

        if hist_seq is not None:
            h = hist_seq.unsqueeze(-1)   # (B,T,1)
            hist_feat = self.hist_proj(h)  # (B,T,hidden)
            q = last_hidden.unsqueeze(1)   # (B,1,hidden)
            attn_out, _w = self.attn(query=q, key=hist_feat, value=hist_feat, need_weights=False)
            attn_out = attn_out.squeeze(1)  # (B, hidden)
            attn_out = self.attn_ln(attn_out + last_hidden)
        else:
            attn_out = last_hidden

        if (cat_inputs is not None and len(cat_inputs)>0) or (cont_inputs is not None and cont_inputs.numel()>0):
            film = self.feat_enc(cat_inputs or {}, cont_inputs)
            g = self.gamma(film)
            b = self.beta(film)
            fused = attn_out * (1.0 + g) + b
        else:
            fused = attn_out

        fused = self.fuse_ln(fused)
        residual = self.decoder(fused)   # (B, out_horizon)
        return residual

# ---------------- Wrapper: base LSTM + adapter ----------------
class LSTMWithAdapter(nn.Module):
    def __init__(self, seq_len, hidden_size=64, num_layers=3, out_dim=28, adapter_kwargs=None):
        super().__init__()
        self.base = LSTMNet(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers, out_dim=out_dim)
        adapter_kwargs = adapter_kwargs or {}
        self.adapter = M5SimpleAdapter(hidden_size=hidden_size, hist_len=adapter_kwargs.get('hist_len', 28),
                                      out_horizon=out_dim,
                                      cat_cardinalities=adapter_kwargs.get('cat_cardinalities', None),
                                      cont_dim=adapter_kwargs.get('cont_dim', 0),
                                      film_dim=adapter_kwargs.get('film_dim', 64),
                                      attn_heads=adapter_kwargs.get('attn_heads', 2),
                                      dropout=adapter_kwargs.get('dropout', 0.1))

    def forward(self, x, hist_seq=None, cat_inputs=None, cont_inputs=None):
        base_pred = self.base(x)  # (B, out_dim)
        out_lstm, (hn, cn) = self.base.lstm(x)
        last = hn[-1]
        residual = self.adapter(last_hidden=last, hist_seq=hist_seq, cat_inputs=cat_inputs, cont_inputs=cont_inputs)
        final = base_pred + residual
        return final, base_pred, residual

# ---------------- Helpers for validation and IO ----------------
def compute_ground_truth_for_val(sales_long, ids, val_days):
    """
    returns (y_true_matrix, ids_kept)
    y_true_matrix shape (n_ids, n_days)
    """
    df_indexed = sales_long.set_index(['id','date'])
    y_true = []
    ids_kept = []
    for _id in ids:
        ok = True
        row = []
        for d in val_days:
            try:
                v = df_indexed.loc[(_id, d)]['sales']
            except Exception:
                ok = False
                break
            row.append(v)
        if ok:
            y_true.append(row)
            ids_kept.append(_id)
    if len(ids_kept) == 0:
        return np.zeros((0,len(val_days)), dtype=np.float32), []
    return np.array(y_true, dtype=np.float32), ids_kept

def evaluate_preds_vs_truth(preds_df, y_true, ids_kept, num_eval_days):
    if len(ids_kept) == 0:
        return None
    preds_sub = preds_df[preds_df['id'].isin(ids_kept)].set_index('id').loc[ids_kept]
    pred_cols = [f'F{h}' for h in range(1, num_eval_days+1)]
    y_pred = preds_sub[pred_cols].values.astype(np.float32)
    mse = ((y_true.reshape(-1) - y_pred.reshape(-1))**2).mean()
    return mse

# ---------------- Training function (uses explicit training samples & validation cutoff) ----------------
def train_lstm_with_adapter(samples_hist_train, sales_long, date_val_cutoff, date_val_end,
                            hist_len=28, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                            cat_feats=None, cont_feats=None, use_log=USE_LOG,
                            adapter_extra_kwargs=None, device=None):
    """
    samples_hist_train: dataframe of training samples (contains hist_1..hist_hist_len and out_1..out_28)
    sales_long: full sales_long df (used to build validation targets)
    date_val_cutoff: cutoff date used for validation (datetime)
    date_val_end: inclusive end date for validation (datetime), e.g. date_d1913
    """
    cat_feats = cat_feats or []
    cont_feats = cont_feats or []
    adapter_extra_kwargs = adapter_extra_kwargs or {}

    # build cat maps from training samples
    cat_maps = build_cat_maps(samples_hist_train, cat_feats)
    cat_cardinalities = {c: (max(v.values())+1 if v else 1) for c, v in cat_maps.items()}

    ds_train = CombinedDataset(samples_hist_train, hist_len=hist_len, cat_feats=cat_feats, cont_feats=cont_feats, cat_maps=cat_maps)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=combined_collate)

    # build validation input (cutoff = date_val_cutoff). include cont/cat
    df_val_hist = build_latest_hist_rows(sales_long, date_val_cutoff, history=hist_len, cat_feats=cat_feats, cont_feats=cont_feats)
    if df_val_hist.shape[0] == 0:
        raise RuntimeError("Validation input (hist) at cutoff %s has zero rows. Check data." % date_val_cutoff)
    ds_val = CombinedDataset(df_val_hist, hist_len=hist_len, cat_feats=cat_feats, cont_feats=cont_feats, cat_maps=cat_maps)
    dl_val = DataLoader(ds_val, batch_size=batch_size*2, shuffle=False, collate_fn=combined_collate)

    # validation days: date_val_cutoff +1 .. date_val_end inclusive
    n_days = int((pd.to_datetime(date_val_end) - pd.to_datetime(date_val_cutoff)).days)
    val_days = [pd.to_datetime(date_val_cutoff) + timedelta(days=i) for i in range(1, n_days+1)]
    num_eval_days = len(val_days)

    adapter_kwargs = dict(hist_len=hist_len, cat_cardinalities=cat_cardinalities, cont_dim=len(cont_feats),
                          film_dim=adapter_extra_kwargs.get('film_dim', ADAPTER_FILMDIM),
                          attn_heads=adapter_extra_kwargs.get('attn_heads', ADAPTER_ATTN_HEADS),
                          dropout=adapter_extra_kwargs.get('dropout', 0.1))

    model = LSTMWithAdapter(seq_len=hist_len, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                            out_dim=OUT_HORIZON, adapter_kwargs=adapter_kwargs)

    if device is None:
        device = torch.device(f'cuda:{GPU_DEVICE_ID}' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_val = float('inf')
    best_state = None

    # Prepare ground truth for validation (for ids present in df_val_hist)
    val_ids = df_val_hist['id'].tolist()
    y_true_val, ids_kept = compute_ground_truth_for_val(sales_long, val_ids, val_days)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for xb_hist, xb_cat, xb_cont, yb in dl_train:
            xb_hist = xb_hist.to(device)
            x_in = xb_hist.unsqueeze(-1)
            xb_cat_dev = {k: v.to(device) for k, v in xb_cat.items()}
            xb_cont = xb_cont.to(device) if xb_cont.numel()>0 else None
            yb = yb.to(device)
            opt.zero_grad()
            preds, base_pred, res = model(x_in, hist_seq=xb_hist, cat_inputs=xb_cat_dev, cont_inputs=xb_cont)
            if use_log:
                loss = F.mse_loss(torch.log1p(preds), torch.log1p(yb))
            else:
                loss = F.mse_loss(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb_hist.size(0)
            n_samples += xb_hist.size(0)
        train_mse = total_loss / (n_samples + 1e-9)

        # validation
        model.eval()
        preds_list = []
        with torch.no_grad():
            for xb_hist, xb_cat, xb_cont, yb in dl_val:
                xb_hist = xb_hist.to(device)
                x_in = xb_hist.unsqueeze(-1)
                xb_cat_dev = {k: v.to(device) for k, v in xb_cat.items()}
                xb_cont = xb_cont.to(device) if xb_cont.numel()>0 else None
                preds, _, _ = model(x_in, hist_seq=xb_hist, cat_inputs=xb_cat_dev, cont_inputs=xb_cont)
                preds_np = preds.cpu().numpy()
                preds_list.append(preds_np)
        if len(preds_list) == 0:
            preds_val_np = np.zeros((0, OUT_HORIZON), dtype=np.float32)
        else:
            preds_val_np = np.vstack(preds_list)
        preds_val_df = pd.DataFrame(preds_val_np, columns=[f'F{h}' for h in range(1,OUT_HORIZON+1)])
        preds_val_df['id'] = df_val_hist['id'].values

        mse = evaluate_preds_vs_truth(preds_val_df, y_true_val, ids_kept, num_eval_days)
        if mse is None:
            print(f"[LSTM+Adapter] Epoch {ep}/{epochs} train_mse={train_mse:.6f} | validation: no overlap")
        else:
            print(f"[LSTM+Adapter] Epoch {ep}/{epochs} train_mse={train_mse:.6f} | val_mse={mse:.6f}")
            if mse < best_val:
                best_val = mse
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                # save validation preds limited to F1..F{num_eval_days}
                save_cols = ['id'] + [f'F{h}' for h in range(1, num_eval_days+1)]
                preds_val_df[save_cols].to_csv(VALIDATION_PRED_CSV, index=False)
                torch.save({'state_dict': best_state}, MODEL_SAVE)
                # save cat_maps for inference
                np.savez_compressed(META_SAVE, **{f"map_{k}": np.array(list(v.items()), dtype=object) for k,v in cat_maps.items()})

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("Warning: no best model was saved during training (best_state is None).")
    return model, cat_maps, best_val

def predict_lstm_with_adapter(model, cat_maps, df_latest_hist, hist_len=28, cat_feats=None, cont_feats=None, device=None):
    cat_feats = cat_feats or []
    cont_feats = cont_feats or []
    hist_cols = [f'hist_{i}' for i in range(1, hist_len+1)]
    X_hist = df_latest_hist[hist_cols].values.astype(np.float32)
    # prepare categorical tensors according to cat_maps for whole batch
    cat_inputs = {}
    for c in cat_feats:
        if c in df_latest_hist.columns:
            cmap = cat_maps.get(c, {})
            vals = df_latest_hist[c].astype(str).fillna('-1').map(lambda x: cmap.get(x, 0)).astype(np.int64).values
            cat_inputs[c] = torch.tensor(vals, dtype=torch.long)
        else:
            cat_inputs[c] = torch.zeros(len(df_latest_hist), dtype=torch.long)
    cont = df_latest_hist[cont_feats].fillna(0).values.astype(np.float32) if cont_feats else np.zeros((len(df_latest_hist), 0), dtype=np.float32)

    if device is None:
        device = torch.device(f'cuda:{GPU_DEVICE_ID}' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        xb_hist = torch.tensor(X_hist, dtype=torch.float32).to(device)
        xb_cat_dev = {k: v.to(device) for k, v in cat_inputs.items()}
        xb_cont = torch.tensor(cont, dtype=torch.float32).to(device) if cont.shape[1] > 0 else None
        x_in = xb_hist.unsqueeze(-1)
        preds, base_pred, residual = model(x_in, hist_seq=xb_hist, cat_inputs=xb_cat_dev, cont_inputs=xb_cont)
        preds_np = preds.cpu().numpy()
        preds_np = np.clip(preds_np, 0.0, None)
        preds_df = pd.DataFrame(preds_np, columns=[f'F{h}' for h in range(1,OUT_HORIZON+1)])
        preds_df['id'] = df_latest_hist['id'].values
        cols = ['id'] + [f'F{h}' for h in range(1,OUT_HORIZON+1)]
        return preds_df[cols]

def make_submission(preds_df, sample_submission_path=SUBMISSION_FILE, out_file=OUTPUT_SUB):
    if not os.path.exists(sample_submission_path):
        preds_df.to_csv(out_file, index=False)
        print('Saved final preds to', out_file, '(no sample_submission found)')
        return out_file
    sub = pd.read_csv(sample_submission_path)
    pred_map = preds_df.set_index('id')
    for col in [c for c in sub.columns if c.startswith('F')]:
        sub[col] = sub['id'].map(lambda x: pred_map.loc[x,col] if x in pred_map.index else 0)
    sub.to_csv(out_file, index=False)
    print('Saved submission to', out_file)
    return out_file

# ----------------- Main flow -----------------
if __name__ == '__main__':
    print('Loading data...')
    sales, calendar, prices = load_data()
    print('sales shape:', sales.shape)
    print('calendar shape:', calendar.shape)
    print('prices shape:', prices.shape)

    # wide->long
    if os.path.exists(OUTPUT_SALES_LONG):
        print('Reading cached sales_long...')
        sales_long = pd.read_feather(OUTPUT_SALES_LONG)
    else:
        if any(c.startswith('d_') for c in sales.columns):
            sales_long = sales_wide_to_long(sales)
        else:
            sales_long = sales.copy()
        sales_long = map_calendar(sales_long, calendar)
        sales_long = merge_prices(sales_long, prices, calendar)
        sales_long = basic_clean(sales_long)
        sales_long = make_lags_rolls_numpy(sales_long)
        sales_long.reset_index(drop=True, inplace=True)
        sales_long.to_feather(OUTPUT_SALES_LONG)
        print(f'Saved {OUTPUT_SALES_LONG}')

    sales_long['date'] = pd.to_datetime(sales_long['date'])
    calendar['date'] = pd.to_datetime(calendar['date'])

    # find date_d1886 and date_d1913
    try:
        date_d1886 = pd.to_datetime(calendar.loc[calendar['d']=='d_1886','date'].iloc[0])
        date_d1913 = pd.to_datetime(calendar.loc[calendar['d']=='d_1913','date'].iloc[0])
    except Exception as e:
        raise RuntimeError("Cannot find d_1886 or d_1913 in calendar.csv") from e
    print('date_d1886:', date_d1886, 'date_d1913:', date_d1913)

    # generate training cutoffs: cutoff s.t. cutoff + OUT_HORIZON <= date_d1886 and cutoff - history exists
    all_dates = sorted(sales_long['date'].unique())
    cutoffs_for_hist = []
    first_date = all_dates[0]
    for d in all_dates:
        d = pd.to_datetime(d)
        if (d - timedelta(days=HIST_LEN)) < pd.to_datetime(first_date):
            continue
        if d + timedelta(days=OUT_HORIZON) <= date_d1886:
            cutoffs_for_hist.append(d)
    print(f"Number of training cutoffs for samples_hist: {len(cutoffs_for_hist)}")

    # generate samples_hist for cutoffs up to date_d1913 - OUT_HORIZON (caching)
    if os.path.exists(OUTPUT_SAMPLES_HIST):
        print('Reading cached samples_hist...')
        samples_hist_all = pd.read_feather(OUTPUT_SAMPLES_HIST)
        samples_hist_train = samples_hist_all[samples_hist_all['cutoff'].isin(cutoffs_for_hist)].copy()
    else:
        cutoffs_gen = [d for d in all_dates if pd.to_datetime(d) + timedelta(days=OUT_HORIZON) <= date_d1913]
        samples_hist_all = generate_samples_with_history(sales_long, cutoffs_gen, horizon=OUT_HORIZON, history=HIST_LEN)
        samples_hist_all.reset_index(drop=True, inplace=True)
        samples_hist_all.to_feather(OUTPUT_SAMPLES_HIST)
        print(f'Saved {OUTPUT_SAMPLES_HIST}')
        samples_hist_train = samples_hist_all[samples_hist_all['cutoff'].isin(cutoffs_for_hist)].copy()

    print('samples_hist_train shape:', samples_hist_train.shape)
    if samples_hist_train.shape[0] == 0:
        raise RuntimeError("No training samples generated for the requested d1..d1886 training split. Check dates and history.")

    # cast categorical for stability
    for c in ['wday', 'month']:
        if c in samples_hist_train.columns:
            samples_hist_train[c] = samples_hist_train[c].astype('category')

    # TRAIN: train_lstm_with_adapter on samples_hist_train, val cutoff = d1886, val end = d1913
    model, cat_maps, best_val = train_lstm_with_adapter(samples_hist_train, sales_long,
                                                        date_val_cutoff=date_d1886,
                                                        date_val_end=date_d1913,
                                                        hist_len=HIST_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                                                        cat_feats=[c for c in CATEGORICAL_FEATURES if c in samples_hist_train.columns],
                                                        cont_feats=[c for c in CONTINUOUS_FEATURES if c in samples_hist_train.columns],
                                                        use_log=USE_LOG,
                                                        adapter_extra_kwargs={'film_dim': ADAPTER_FILMDIM, 'attn_heads': ADAPTER_ATTN_HEADS},
                                                        device=torch.device(f'cuda:{GPU_DEVICE_ID}' if (USE_GPU and torch.cuda.is_available()) else 'cpu'))
    print('Training finished. Best validation MSE:', best_val)
    print('Best validation preds saved to', VALIDATION_PRED_CSV)
    print('Best model saved to', MODEL_SAVE)
    print('Cat maps saved to', META_SAVE)

    # FINAL PREDICTION: use cutoff = d1913 -> predict d1914..d1941
    final_cutoff = date_d1913
    df_final_hist = build_latest_hist_rows(sales_long, final_cutoff, history=HIST_LEN,
                                           cat_feats=[c for c in CATEGORICAL_FEATURES if c in sales_long.columns],
                                           cont_feats=[c for c in CONTINUOUS_FEATURES if c in sales_long.columns])
    print('df_final_hist rows:', df_final_hist.shape[0])
    if df_final_hist.shape[0] == 0:
        raise RuntimeError("No final prediction rows (missing full history for cutoff d1913).")

    preds_final = predict_lstm_with_adapter(model, cat_maps, df_final_hist, hist_len=HIST_LEN,
                                           cat_feats=[c for c in CATEGORICAL_FEATURES if c in df_final_hist.columns],
                                           cont_feats=[c for c in CONTINUOUS_FEATURES if c in df_final_hist.columns],
                                           device=torch.device(f'cuda:{GPU_DEVICE_ID}' if (USE_GPU and torch.cuda.is_available()) else 'cpu'))
    preds_final.to_csv(FINAL_PRED_CSV, index=False)
    print('Saved final raw preds to', FINAL_PRED_CSV)

    # if sample_submission provided, align and save OUTPUT_SUB
    if os.path.exists(SUBMISSION_FILE):
        make_submission(preds_final, sample_submission_path=SUBMISSION_FILE, out_file=OUTPUT_SUB)
    else:
        print('sample_submission not found; only raw final preds saved.')

    print('Done.')
