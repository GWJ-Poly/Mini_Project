
import os
import gc
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
import random
import warnings
warnings.filterwarnings('ignore')

# PyTorch for MLP / LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------- user config -----------------
DATA_DIR = './data'  
SALES_FILE = os.path.join(DATA_DIR, 'sales_train_validation.csv')
CALENDAR_FILE = os.path.join(DATA_DIR, 'calendar.csv')
PRICES_FILE = os.path.join(DATA_DIR, 'sell_prices.csv')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

OUTPUT_SALES_LONG = 'sales_long.feather'
OUTPUT_SAMPLES = 'samples.feather'
OUTPUT_SAMPLES_HIST = 'samples_hist.feather'  
OUTPUT_SUB = 'b2_submission_m5_dt.csv'
MODEL_SAVE = 'dt.pkl' 
TORCH_SAVE = 'dt.pt'     
VALIDATION_PRED_CSV = 'b2_validation_preds_best_dt.csv'
FINAL_PRED_CSV = 'b2_submission_preds_final_dt.csv'

# GPU or not
USE_GPU = True
GPU_DEVICE_ID = 2

# LightGBM num rounds
NUM_BOOST_ROUND = 1000

# choose models： 'lightgbm','mlp','lstm','linear','lasso','dt','svr'
SELECTED_MODEL = 'dt'

# PyTorch training params (for MLP/LSTM)
PT_EPOCHS = 100
PT_BATCH_SIZE = 256
PT_LR = 1e-4
# --------------------------------------------


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def load_data():
    print('Loading data...')
    sales = pd.read_csv(SALES_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)
    prices = pd.read_csv(PRICES_FILE)
    print('sales shape:', sales.shape)
    print('calendar shape:', calendar.shape)
    print('prices shape:', prices.shape)
    return sales, calendar, prices

def sales_wide_to_long(sales_wide):
    id_cols = [c for c in sales_wide.columns if c.startswith('id') or c in ['item_id','dept_id','cat_id','store_id','state_id']]
    date_cols = [c for c in sales_wide.columns if c.startswith('d_')]
    print('Found %d id cols, %d date cols' % (len(id_cols), len(date_cols)))
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
        price_arr = df_item['sell_price'].values.astype(float) if 'sell_price' in df_item.columns else np.zeros_like(sales_arr)
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
        hist_dates = [cur_date - timedelta(days=i) for i in range(history, 0, -1)]  # oldest -> newest
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

def cast_categorical(X_df, categorical_feats):
    X = X_df.copy()
    for c in categorical_feats:
        if c in X.columns:
            X[c] = X[c].astype('category')
    return X

# ---------------- Helpers: prepare features for sklearn / torch ----------------
def prepare_X_for_sklearn(df_X, categorical_feats):
    X = df_X.copy()
    for c in categorical_feats:
        if c in X.columns:
            if pd.api.types.is_categorical_dtype(X[c]):
                X[c] = X[c].cat.codes
            else:
                X[c] = X[c].astype('int').fillna(0)
    X = X.fillna(0)
    return X.values

def prepare_X_for_torch(df_X, categorical_feats):
    X = df_X.copy()
    for c in categorical_feats:
        if c in X.columns:
            if pd.api.types.is_categorical_dtype(X[c]):
                X[c] = X[c].cat.codes
            else:
                X[c] = X[c].astype('int').fillna(0)
    X = X.fillna(0)
    return X.values.astype(np.float32)

# ---------------- sklearn model training wrappers ----------------
def train_sklearn_multi(samples_df, feature_cols, categorical_feats, model_type='linear'):
    X_df = samples_df[feature_cols].copy()
    X = prepare_X_for_sklearn(X_df, categorical_feats)
    y = samples_df[[f'out_{h}' for h in range(1,29)]].values.astype(np.float32)

    if model_type == 'linear':
        base = LinearRegression()
    elif model_type == 'lasso':
        base = Lasso(alpha=0.1, max_iter=2000)
    elif model_type == 'dt':
        base = DecisionTreeRegressor(max_depth=20)
    elif model_type == 'svr':
        base = SVR(kernel='rbf', C=1.0)
    elif model_type == 'lightgbm':
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',   
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 32,
            'max_depth': 5,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'lambda_l1': 0.0,
            'lambda_l2': 0.1,
            'verbose': -1
        }
        if USE_GPU:
            lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': GPU_DEVICE_ID})
        base = lgb.LGBMRegressor(**lgb_params, n_estimators=NUM_BOOST_ROUND)
    else:
        raise ValueError('Unknown sklearn model_type')

    multi = MultiOutputRegressor(base, n_jobs=-1 if not USE_GPU else 1)
    print(f"Training sklearn model {model_type} ...")
    multi.fit(X, y)
    print('Training finished.')
    return multi

class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[256,128,64,32]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev,h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

class LSTMNet(nn.Module):
    def __init__(self, seq_len, hidden_size=128, num_layers=3, out_dim=28):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, out_dim)
        )
    def forward(self,x):  # x: (batch, seq_len, 1)
        out, (hn, cn) = self.lstm(x)
        last = hn[-1]  # (batch, hidden)
        return self.fc(last)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_validation_targets(sales_long, ids, val_days):
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
        return np.zeros((0, len(val_days))), []
    return np.array(y_true, dtype=np.float32), ids_kept

def evaluate_preds_vs_truth(preds_df, y_true, ids_kept, num_eval_days):
    """
    preds_df: columns id,F1..F28
    y_true: numpy (n_ids, num_eval_days)
    ids_kept: list of ids
    """
    if len(ids_kept) == 0:
        return None
    preds_sub = preds_df[preds_df['id'].isin(ids_kept)].set_index('id').loc[ids_kept]
    pred_cols = [f'F{h}' for h in range(1, num_eval_days+1)]
    y_pred = preds_sub[pred_cols].values.astype(np.float32)
    mse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
    return mse

def train_mlp_torch(samples_df, feature_cols, categorical_feats, df_val, df_val_hist, val_days, sales_long,
                    epochs=PT_EPOCHS, batch_size=PT_BATCH_SIZE, lr=PT_LR):
    # Prepare train
    X_np = prepare_X_for_torch(samples_df[feature_cols], categorical_feats)
    y_np = samples_df[[f'out_{h}' for h in range(1,29)]].values.astype(np.float32)
    device = torch.device('cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
    model = MLPNet(in_dim=X_np.shape[1], out_dim=28).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = SimpleDataset(X_np, y_np)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # validation ground truth
    num_eval_days = len(val_days)
    val_ids_all = df_val['id'].values.tolist()
    y_true_val, ids_kept = compute_validation_targets(sales_long, val_ids_all, val_days)

    best_mse = float('inf')
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(ds)
        # validation
        model.eval()
        with torch.no_grad():
            # prepare validation input X (df_val)
            X_val = prepare_X_for_torch(df_val[feature_cols], categorical_feats)
            xb_val = torch.from_numpy(X_val).to(device)
            pred_val = model(xb_val).cpu().numpy()
        preds_val_df = pd.DataFrame(pred_val, columns=[f'F{h}' for h in range(1,29)])
        preds_val_df['id'] = df_val['id'].values
        mse = evaluate_preds_vs_truth(preds_val_df, y_true_val, ids_kept, num_eval_days)
        if mse is None:
            print(f"[MLP] Epoch {ep+1}/{epochs} train_mse={avg:.6f} | validation: no overlap")
        else:
            print(f"[MLP] Epoch {ep+1}/{epochs} train_mse={avg:.6f} | val_mse={mse:.6f}")
            if mse < best_mse:
                best_mse = mse
                best_state = model.state_dict()
                # save preds CSV for best
                preds_val_df.to_csv(VALIDATION_PRED_CSV, index=False)
                torch.save(best_state, TORCH_SAVE)
    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_mse

def train_lstm_torch(samples_hist_df, hist_len=28, df_val_hist=None, val_days=None, sales_long=None,
                     epochs=PT_EPOCHS, batch_size=PT_BATCH_SIZE, lr=PT_LR):
    hist_cols = [f'hist_{i}' for i in range(1, hist_len+1)]
    X_hist = samples_hist_df[hist_cols].values.astype(np.float32)
    X_seq = X_hist.reshape((-1, hist_len, 1))
    y_np = samples_hist_df[[f'out_{h}' for h in range(1,29)]].values.astype(np.float32)

    device = torch.device('cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
    model = LSTMNet(seq_len=hist_len, hidden_size=128, num_layers=3, out_dim=28).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = SimpleDataset(X_seq, y_np)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # validation ground truth (df_val_hist contains id + hist cols)
    num_eval_days = len(val_days)
    val_ids_all = df_val_hist['id'].values.tolist()
    y_true_val, ids_kept = compute_validation_targets(sales_long, val_ids_all, val_days)

    best_mse = float('inf')
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device).float()
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(ds)

        # validation
        model.eval()
        with torch.no_grad():
            X_val = df_val_hist[[f'hist_{i}' for i in range(1, hist_len+1)]].values.astype(np.float32).reshape(-1, hist_len, 1)
            xb_val = torch.from_numpy(X_val).to(device)
            pred_val = model(xb_val).cpu().numpy()
        preds_val_df = pd.DataFrame(pred_val, columns=[f'F{h}' for h in range(1,29)])
        preds_val_df['id'] = df_val_hist['id'].values
        mse = evaluate_preds_vs_truth(preds_val_df, y_true_val, ids_kept, num_eval_days)
        if mse is None:
            print(f"[LSTM] Epoch {ep+1}/{epochs} train_mse={avg:.6f} | validation: no overlap")
        else:
            print(f"[LSTM] Epoch {ep+1}/{epochs} train_mse={avg:.6f} | val_mse={mse:.6f}")
            if mse < best_mse:
                best_mse = mse
                best_state = model.state_dict()
                preds_val_df.to_csv(VALIDATION_PRED_CSV, index=False)
                torch.save(best_state, TORCH_SAVE)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_mse

# ---------------- Predict wrappers ----------------
def predict_with_sklearn(model, df_latest, feature_cols, categorical_feats):
    Xp = df_latest[feature_cols].copy()
    Xp_proc = prepare_X_for_sklearn(Xp, categorical_feats)
    preds = model.predict(Xp_proc)  # (n,28)
    preds_df = pd.DataFrame(preds, columns=[f'F{h}' for h in range(1,29)])
    preds_df['id'] = df_latest['id'].values
    cols = ['id'] + [f'F{h}' for h in range(1,29)]
    return preds_df[cols]

def predict_with_mlp(model, df_latest, feature_cols, categorical_feats):
    Xp = df_latest[feature_cols].copy()
    Xp_np = prepare_X_for_torch(Xp, categorical_feats)
    device = torch.device('cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xp_np).to(device)
        pred = model(xb).cpu().numpy()
    preds_df = pd.DataFrame(pred, columns=[f'F{h}' for h in range(1,29)])
    preds_df['id'] = df_latest['id'].values
    cols = ['id'] + [f'F{h}' for h in range(1,29)]
    return preds_df[cols]

def predict_with_lstm(model, df_latest_hist, hist_len=28):
    hist_cols = [f'hist_{i}' for i in range(1, hist_len+1)]
    X_hist = df_latest_hist[hist_cols].values.astype(np.float32).reshape(-1, hist_len, 1)
    device = torch.device('cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_hist).to(device)
        pred = model(xb).cpu().numpy()
    preds_df = pd.DataFrame(pred, columns=[f'F{h}' for h in range(1,29)])
    preds_df['id'] = df_latest_hist['id'].values
    cols = ['id'] + [f'F{h}' for h in range(1,29)]
    return preds_df[cols]

# ---------------- submission helper ----------------
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

# ------------------ Helpers to build latest input rows for arbitrary cutoff --------------
def build_latest_input_rows(sales_long_df, cutoff_date, feature_cols, categorical_feats):
    df_indexed = sales_long_df.set_index(['id','date'])
    ids = sales_long_df['id'].unique()
    rows = []
    for _id in ids:
        try:
            base = df_indexed.loc[(_id, pd.to_datetime(cutoff_date))]
        except KeyError:
            continue
        row = {'id': _id}
        for f in feature_cols:
            row[f] = base.get(f, np.nan)
        for c in categorical_feats:
            if c not in row and c in base:
                row[c] = base.get(c, np.nan)
        rows.append(row)
    df_rows = pd.DataFrame(rows)
    for c in categorical_feats:
        if c in df_rows.columns:
            df_rows[c] = df_rows[c].astype('category')
    return df_rows

def build_latest_hist_rows(sales_long_df, cutoff_date, history=28):
    df_indexed = sales_long_df.set_index(['id','date'])
    ids = sales_long_df['id'].unique()
    rows = []
    hdates = [pd.to_datetime(cutoff_date) - timedelta(days=i) for i in range(history,0,-1)]
    for _id in ids:
        ok = True
        row = {'id': _id}
        for i, hd in enumerate(hdates, start=1):
            try:
                row[f'hist_{i}'] = df_indexed.loc[(_id, hd)]['sales']
            except Exception:
                ok = False
                break
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    sales, calendar, prices = load_data()

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

    try:
        date_d1886 = pd.to_datetime(calendar.loc[calendar['d'] == 'd_1886', 'date'].iloc[0])
        date_d1913 = pd.to_datetime(calendar.loc[calendar['d'] == 'd_1913', 'date'].iloc[0])
    except Exception as e:
        raise RuntimeError("无法在 calendar.csv 中找到 d_1886 或 d_1913，请检查 calendar 文件.") from e

    print('date_d1886:', date_d1886, 'date_d1913:', date_d1913)

    horizon = 28
    all_candidate_cutoffs = sorted(list(set(sales_long['date'])))
    cutoffs_for_generation = [d for d in all_candidate_cutoffs if pd.to_datetime(d) <= (sales_long['date'].max() - timedelta(days=horizon))]
    if os.path.exists(OUTPUT_SAMPLES):
        print('Reading cached samples...')
        samples = pd.read_feather(OUTPUT_SAMPLES)
    else:
        samples = generate_samples(sales_long, cutoffs_for_generation, horizon=horizon)
        samples.reset_index(drop=True, inplace=True)
        samples.to_feather(OUTPUT_SAMPLES)
        print(f'Saved {OUTPUT_SAMPLES}')
    print('samples shape:', samples.shape)

    # LSTM samples_hist
    if SELECTED_MODEL == 'lstm':
        if os.path.exists(OUTPUT_SAMPLES_HIST):
            samples_hist = pd.read_feather(OUTPUT_SAMPLES_HIST)
        else:
            samples_hist = generate_samples_with_history(sales_long, cutoffs_for_generation, horizon=horizon, history=28)
            samples_hist.reset_index(drop=True, inplace=True)
            samples_hist.to_feather(OUTPUT_SAMPLES_HIST)
            print(f'Saved {OUTPUT_SAMPLES_HIST}')
        print('samples_hist shape:', samples_hist.shape)
    else:
        samples_hist = None

    feature_cols = ['lag_1','lag_7','lag_28','roll_mean_7','roll_mean_28','roll_std_7',
                    'sell_price','price_lag_7','wday','month','is_weekend']
    categorical_feats = ['wday','month']

    for c in categorical_feats:
        if c in samples.columns:
            samples[c] = samples[c].astype('category')
        if samples_hist is not None and c in samples_hist.columns:
            samples_hist[c] = samples_hist[c].astype('category')

    samples['cutoff'] = pd.to_datetime(samples['cutoff'])
    samples_train = samples[samples['cutoff'] + timedelta(days=horizon) <= date_d1886].copy()
    print('Train samples shape (cutoff + 28 <= d1886):', samples_train.shape)
    if samples_hist is not None:
        samples_hist['cutoff'] = pd.to_datetime(samples_hist['cutoff'])
        samples_hist_train = samples_hist[samples_hist['cutoff'] + timedelta(days=horizon) <= date_d1886].copy()
        print('samples_hist_train shape:', samples_hist_train.shape)
    else:
        samples_hist_train = None

    if len(samples_train) < 100:
        print('Warning: 训练样本数量较少:', len(samples_train))

    val_cutoff = date_d1886
    df_val_X = build_latest_input_rows(sales_long, val_cutoff, feature_cols, categorical_feats)
    if SELECTED_MODEL == 'lstm':
        df_val_hist = build_latest_hist_rows(sales_long, val_cutoff, history=28)
        df_val_hist = df_val_hist.merge(df_val_X[['id']], on='id', how='inner')
    else:
        df_val_hist = None

    # 验证目标天列表 d1887..d1913 (inclusive)
    val_days = [(val_cutoff + timedelta(days=i)) for i in range(1, (date_d1913 - val_cutoff).days + 1)]
    num_eval_days = len(val_days)
    print('Validation days:', val_days[0], '->', val_days[-1], f'({num_eval_days} days)')

    # 7. 训练并在验证集上每 epoch 评估保存最佳模型
    trained = None
    torch_model = None
    best_val_mse = None

    if SELECTED_MODEL in ['linear','lasso','dt','svr','lightgbm']:
        trained = train_sklearn_multi(samples_train, feature_cols, categorical_feats, model_type=SELECTED_MODEL if SELECTED_MODEL!='lightgbm' else 'lightgbm')
        if SELECTED_MODEL == 'lightgbm' or SELECTED_MODEL in ['linear','lasso','dt','svr']:
            df_val = df_val_X
            preds_val = predict_with_sklearn(trained, df_val, feature_cols, categorical_feats)
            y_true_val, ids_kept = compute_validation_targets(sales_long, df_val['id'].tolist(), val_days)
            best_val_mse = evaluate_preds_vs_truth(preds_val, y_true_val, ids_kept, num_eval_days)
            print('Validation MSE (sklearn model) =', best_val_mse)
            preds_val.to_csv(VALIDATION_PRED_CSV, index=False)
            joblib.dump(trained, MODEL_SAVE)
            print('Saved sklearn model to', MODEL_SAVE)
    elif SELECTED_MODEL == 'mlp':
        torch_model, best_val_mse = train_mlp_torch(samples_train, feature_cols, categorical_feats,
                                                    df_val_X, df_val_hist, val_days, sales_long,
                                                    epochs=PT_EPOCHS, batch_size=PT_BATCH_SIZE, lr=PT_LR)
        print('Best validation MSE (MLP):', best_val_mse)
    elif SELECTED_MODEL == 'lstm':
        if samples_hist_train is None or samples_hist_train.shape[0] == 0:
            raise RuntimeError('samples_hist_train is required for LSTM training but not generated or empty.')
        # train_lstm_torch expects samples_hist_train and df_val_hist
        torch_model, best_val_mse = train_lstm_torch(samples_hist_train, hist_len=28,
                                                     df_val_hist=df_val_hist, val_days=val_days,
                                                     sales_long=sales_long,
                                                     epochs=PT_EPOCHS, batch_size=PT_BATCH_SIZE, lr=PT_LR)
        print('Best validation MSE (LSTM):', best_val_mse)
    else:
        raise ValueError('Unknown SELECTED_MODEL')

    final_cutoff = date_d1913
    print('Building final input for cutoff:', final_cutoff)
    df_final_X = build_latest_input_rows(sales_long, final_cutoff, feature_cols, categorical_feats)
    if SELECTED_MODEL == 'lstm':
        df_final_hist = build_latest_hist_rows(sales_long, final_cutoff, history=28)
        df_final_hist = df_final_hist.merge(df_final_X[['id']], on='id', how='inner')
        preds_final = predict_with_lstm(torch_model, df_final_hist, hist_len=28)
    elif SELECTED_MODEL == 'mlp':
        preds_final = predict_with_mlp(torch_model, df_final_X, feature_cols, categorical_feats)
    else:
        preds_final = predict_with_sklearn(trained, df_final_X, feature_cols, categorical_feats)

    preds_final.to_csv(FINAL_PRED_CSV, index=False)
    print('Saved final raw preds to', FINAL_PRED_CSV)
    if os.path.exists(SUBMISSION_FILE):
        make_submission(preds_final, sample_submission_path=SUBMISSION_FILE, out_file=OUTPUT_SUB)

    print('Done. Model:', SELECTED_MODEL)
    print('Best validation MSE:', best_val_mse)
