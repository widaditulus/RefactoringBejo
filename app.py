# = =========================================================================
# ## NAMA FILE: app.py
# ## STATUS: VERSI 10.6 (FINAL VECTORIZED REFACTOR)
# ## PERBAIKAN:
# ## 1. Implementasi "vectorized" penuh pada generate_all_features.
# ## 2. Menghilangkan pembuatan pd.Series di dalam loop yang menjadi akar masalah performa.
# ## 3. Ini adalah perbaikan definitif untuk masalah kecepatan training.
# =========================================================================

import os
import joblib
import requests
import numpy as np
import pandas as pd
import logging
import json
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
from itertools import combinations
import signal
import sys
import uuid
import threading
import re

MODEL_VERSION = "10.1" # Versi arsitektur model tidak berubah

# =================================================================
# ## KONFIGURASI APLIKASI
# =================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder='.')
MODELS_DIR, LOGS_DIR, DATA_DIR, CONFIG_DIR = (os.path.join(BASE_DIR, d) for d in ["models", "logs", "data", "config"])
for d in [MODELS_DIR, LOGS_DIR, DATA_DIR, CONFIG_DIR]: os.makedirs(d, exist_ok=True)
log_file = os.path.join(LOGS_DIR, 'app.log')
handler = RotatingFileHandler(log_file, maxBytes=1048576, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info("Aplikasi Dimulai.")
evaluation_jobs, weight_adjustment_jobs = {}, {}
data_cache, feature_cache = {}, {}
shared_jobs_lock, shared_weights_lock = threading.Lock(), threading.Lock()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app_database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
WEIGHTS_CONFIG_PATH = os.path.join(CONFIG_DIR, 'weights_config.json')
posisi_weights = {}

def load_weights():
    global posisi_weights
    try:
        with open(WEIGHTS_CONFIG_PATH, 'r') as f:
            posisi_weights = json.load(f)
        app.logger.info("Bobot dinamis berhasil dimuat.")
    except Exception as e:
        app.logger.error(f"Gagal memuat file bobot: {e}")
        posisi_weights = {}

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tanggal, pasaran = db.Column(db.String(10), nullable=False), db.Column(db.String(50), nullable=False)
    prediksi_cb, am = db.Column(db.Integer, nullable=False), db.Column(db.String(50), nullable=True)
    prediksi_as, prediksi_kop = db.Column(db.String(50), nullable=True), db.Column(db.String(50), nullable=True)
    prediksi_kepala, prediksi_ekor = db.Column(db.String(50), nullable=True), db.Column(db.String(50), nullable=True)
    hasil, status, cb_status = db.Column(db.String(20), default='-'), db.Column(db.String(20), default='Menunggu'), db.Column(db.String(10), default='...')
    __table_args__ = (db.UniqueConstraint('tanggal', 'pasaran', name='_tanggal_pasaran_uc'),)

with app.app_context(): db.create_all()

DATA_URLS = {
    "china": "https://raw.githubusercontent.com/widaditulus/4D/main/china_data.csv",
    "hk": "https://raw.githubusercontent.com/widaditulus/4D/main/hk_data.csv",
    "magnum": "https://raw.githubusercontent.com/widaditulus/4D/main/magnum_data.csv",
    "sgp": "https://raw.githubusercontent.com/widaditulus/4D/main/sgp_data.csv",
    "sydney": "https://raw.githubusercontent.com/widaditulus/4D/main/sydney_data.csv",
    "taiwan": "https://raw.githubusercontent.com/widaditulus/4D/main/taiwan_data.csv",
}
MIN_HISTORICAL_DATA = 181 # Disesuaikan dengan window rolling terbesar + 1

# =================================================================
# ## FUNGSI HELPER & DATA
# =================================================================
def _get_remote_last_modified(url: str):
    try:
        response = requests.head(url, timeout=10)
        response.raise_for_status()
        last_modified_str = response.headers.get('Last-Modified')
        if last_modified_str: return parsedate_to_datetime(last_modified_str)
    except requests.exceptions.RequestException: return None

def _download_and_save(url: str, local_path: str):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(local_path, 'w', encoding='utf-8') as f: f.write(response.text)
        return True, None
    except requests.exceptions.RequestException as e:
        return False, f"Gagal mengunduh data dari {url}: {e}"

def get_data(pasaran: str, force_reload: bool = False):
    url = DATA_URLS.get(pasaran)
    local_path = os.path.join(DATA_DIR, f"{pasaran}_data.csv")
    reloaded = False
    if pasaran in data_cache and not force_reload: return data_cache[pasaran].copy(), None, reloaded
    should_download = False
    if force_reload or not os.path.exists(local_path):
        should_download = True
    else:
        try:
            local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
            remote_mtime = _get_remote_last_modified(url)
            if remote_mtime and remote_mtime > local_mtime: should_download = True
        except Exception: should_download = True
    if should_download:
        success, error = _download_and_save(url, local_path)
        reloaded = success
        if not success:
            if os.path.exists(local_path): df = pd.read_csv(local_path)
            else: return None, error, reloaded
        else: df = pd.read_csv(local_path)
    else: df = pd.read_csv(local_path)
    try:
        df = df.dropna(subset=['date', 'result'])
        for col in ['as', 'kop', 'kepala', 'ekor']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['as', 'kop', 'kepala', 'ekor'])
        df = df.astype({'as': int, 'kop': int, 'kepala': int, 'ekor': int})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        data_cache[pasaran] = df.copy()
        return df, None, reloaded
    except Exception as e: return None, f"Gagal memproses file data: {e}", reloaded

# =================================================================
# ## FUNGSI FEATURE ENGINEERING & PREDIKSI (VERSI VEKTORISASI FINAL)
# =================================================================
def generate_all_features(df, pasaran):
    """
    Versi Vektorisasi Penuh: Menghindari pembuatan Series di dalam loop
    untuk performa maksimal.
    """
    features = {}
    window = 180

    # OPTIMISASI: Hitung semua kemunculan digit sebagai numpy array HANYA SEKALI.
    digits_arr = df[['as', 'kop', 'kepala', 'ekor']].values
    digit_cols_np = {i: np.any(digits_arr == i, axis=1).astype(int) for i in range(10)}

    # OPTIMISASI: Buat DataFrame dari semua digit HANYA SEKALI, lalu lakukan rolling secara "borongan".
    digit_df = pd.DataFrame(digit_cols_np)

    # 1. & 4. Fitur frekuensi dan tren (rolling) secara borongan
    rolling_sum_w = digit_df.rolling(window=window, min_periods=1).sum().shift(1)
    rolling_mean_7 = digit_df.rolling(window=7, min_periods=1).mean().shift(1)
    rolling_mean_30 = digit_df.rolling(window=30, min_periods=1).mean().shift(1)
    rolling_mean_90 = digit_df.rolling(window=90, min_periods=1).mean().shift(1)

    for i in range(10):
        features[f'freq_total_{i}_roll{window}'] = rolling_sum_w[i].values
        features[f'trend_{i}_roll{7}'] = rolling_mean_7[i].values
        features[f'trend_{i}_roll{30}'] = rolling_mean_30[i].values
        features[f'trend_{i}_roll{90}'] = rolling_mean_90[i].values

    # 3. Fitur frekuensi posisi
    for pos in ['as', 'kop', 'kepala', 'ekor']:
        pos_dummies = pd.get_dummies(df[pos], prefix=f'freq_{pos}').reindex(columns=[f'freq_{pos}_{i}' for i in range(10)], fill_value=0)
        rolling_sum = pos_dummies.rolling(window=window, min_periods=1).sum().shift(1)
        for col in rolling_sum.columns:
            features[col] = rolling_sum[col].values

    # 5. Fitur "cold score"
    # OPTIMISASI: Gunakan numpy array secara langsung, hindari membuat Series baru.
    row_num_s = pd.Series(np.arange(len(df)))
    for i in range(10):
        seen_at = row_num_s.where(digit_cols_np[i] == 1)
        last_seen = seen_at.ffill().fillna(-1)
        features[f'cold_score_{i}'] = (row_num_s - last_seen).shift(1).fillna(window).values

    # 6. Fitur kemunculan pasangan digit (Co-occurrence)
    # OPTIMISASI: Hitung semua pair sebagai numpy array, buat DataFrame HANYA SEKALI, lalu rolling borongan.
    pair_cols_np = {}
    for d1, d2 in combinations(range(10), 2):
        pair_cols_np[f'pair_prob_{d1}{d2}'] = digit_cols_np[d1] & digit_cols_np[d2]
    
    pair_df = pd.DataFrame(pair_cols_np)
    rolling_pair_mean = pair_df.rolling(window=window, min_periods=1).mean().shift(1)
    for col in rolling_pair_mean.columns:
        features[f'{col}_roll{window}'] = rolling_pair_mean[col].values

    # 7. Fitur berbasis waktu (musiman)
    dt = df['date'].dt
    features['dayofweek_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
    features['dayofweek_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
    features['dayofyear_sin'] = np.sin(2 * np.pi * dt.dayofyear / 366)
    features['dayofyear_cos'] = np.cos(2 * np.pi * dt.dayofyear / 366)

    # Buat DataFrame dari dictionary di akhir. Ini sangat cepat.
    X_df = pd.DataFrame(features)
    X_df.fillna(0, inplace=True)
    return X_df

def get_or_generate_features(pasaran, df):
    cache_key = f"{pasaran}_{len(df)}"
    if cache_key in feature_cache: return feature_cache[cache_key].copy()
    
    X_df = generate_all_features(df, pasaran)
    feature_cache[cache_key] = X_df.copy()
    return X_df

def create_features_and_targets(df, pasaran):
    target_cols = ['as', 'kop', 'kepala', 'ekor']
    df.dropna(subset=target_cols, inplace=True); df = df.astype({col: int for col in target_cols})
    
    X_df = get_or_generate_features(pasaran, df.copy())
    
    valid_indices = X_df.dropna().index
    X = X_df.loc[valid_indices].values
    
    y_pos_targets = {pos: df.loc[valid_indices, pos].values for pos in target_cols}
    y_cb_targets = np.zeros((len(valid_indices), 10), dtype=int)
    for i, index in enumerate(valid_indices):
        digits_in_result = set(df.loc[index, target_cols].values)
        for digit in digits_in_result:
            y_cb_targets[i, digit] = 1
            
    return X, y_pos_targets, y_cb_targets

def get_full_prediction(df_historis, models, scaler, target_date, pasaran):
    future_row = pd.DataFrame([{'date': target_date}])
    df_for_generation = pd.concat([df_historis, future_row], ignore_index=True)
    
    X_df_full = generate_all_features(df_for_generation.copy(), pasaran)

    last_features_row = X_df_full.iloc[-1:]
    if last_features_row.isnull().values.any():
        app.logger.error(f"Fitur NaN terdeteksi untuk prediksi tanggal {target_date}: {last_features_row}")
        raise ValueError("Fitur yang dihasilkan mengandung NaN, tidak bisa melanjutkan prediksi.")
        
    input_vector = last_features_row.values.reshape(1, -1)
    if input_vector.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Jumlah fitur tidak cocok. Model mengharapkan {scaler.n_features_in_}, tetapi mendapat {input_vector.shape[1]}.")

    input_scaled = scaler.transform(input_vector)
    
    pos_models = models['pos_models']
    pos_probas = {pos: pos_models[pos].predict_proba(input_scaled)[0] for pos in ['as', 'kop', 'kepala', 'ekor']}
    meta_model = models['meta_model']
    meta_features = np.concatenate(list(pos_probas.values())).reshape(1, -1)
    skor_ai_cb_am = meta_model.predict_proba(meta_features)[0]
    skor_final_cb_am = sorted(enumerate(skor_ai_cb_am), key=lambda x: x[1], reverse=True)
    cb = int(skor_final_cb_am[0][0])
    am_candidates = [int(item[0]) for item in skor_final_cb_am[:4]]
    posisi = {}
    with shared_weights_lock: pasaran_weights = posisi_weights.get(pasaran, {})
    for pos in ['as', 'kop', 'kepala', 'ekor']:
        frekuensi_pos = df_historis[pos].value_counts(normalize=True).reindex(range(10), fill_value=0)
        weights = pasaran_weights.get(pos, {"ai": 0.5, "freq": 0.5})
        pos_scores = [(d, (weights["ai"] * pos_probas[pos][d]) + (weights["freq"] * frekuensi_pos.get(d, 0))) for d in range(10)]
        posisi[pos] = [int(item[0]) for item in sorted(pos_scores, key=lambda x: x[1], reverse=True)[:3]]
    return {"cb": cb, "am": sorted(am_candidates), "posisi": posisi, "pos_probas": pos_probas}

# =================================================================
# ## ENDPOINTS FLASK (API)
# =================================================================

def find_latest_model(pasaran):
    pattern = re.compile(f"^{pasaran}_model_v(\\d+\\.\\d+)\\.joblib$")
    latest_version, latest_model_path = -1.0, None
    for filename in os.listdir(MODELS_DIR):
        match = pattern.match(filename)
        if match:
            version = float(match.group(1))
            if version > latest_version:
                latest_version, latest_model_path = version, os.path.join(MODELS_DIR, filename)
    return latest_model_path

@app.route('/')
def index(): return render_template('index.html')

@app.route('/main_refactored.js')
def serve_js(): return send_from_directory('.', 'main_refactored.js')

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        pasaran = data['pasaran']
        config_payload = data.get('config', {})
        
        df, error, reloaded = get_data(pasaran, force_reload=True)
        if error: return jsonify({"error": error}), 500
        
        if reloaded:
            keys_to_del = [k for k in feature_cache if k.startswith(pasaran)]
            for k in keys_to_del: del feature_cache[k]
        
        X, y_pos_targets, y_cb_targets = create_features_and_targets(df.copy(), pasaran)
        if X.shape[0] < MIN_HISTORICAL_DATA: return jsonify({"error": f"Data historis tidak cukup (min {MIN_HISTORICAL_DATA} hari)."}), 400
        
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        kf = KFold(n_splits=5, shuffle=False)
        meta_features = np.zeros((X.shape[0], 40))
        app.logger.info("Memulai pembuatan meta-features dengan Cross-Validation...")
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            fold_meta_features = []
            for pos in ['as', 'kop', 'kepala', 'ekor']:
                y_pos_train_fold = y_pos_targets[pos][train_idx]
                model = LogisticRegression(random_state=42, solver='liblinear', max_iter=100)
                model.fit(X_train_fold, y_pos_train_fold)
                fold_meta_features.append(model.predict_proba(X_val_fold))
            meta_features[val_idx] = np.hstack(fold_meta_features)
        
        app.logger.info("Selesai membuat meta-features. Melatih model final...")

        trained_pos_models = {}
        for pos in ['as', 'kop', 'kepala', 'ekor']:
            final_config = config_payload.get('global', {}).copy()
            final_config.update({k:v for k,v in config_payload.get('pos_specific',{}).get(pos,{}).items() if v is not None})
            
            model = MLPClassifier(
                hidden_layer_sizes=(final_config.get('h1', 32), final_config.get('h2', 16)), 
                max_iter=final_config.get('epochs', 100), 
                random_state=42, 
                early_stopping=True, 
                n_iter_no_change=final_config.get('patience', 15), 
                shuffle=False
            )
            model.fit(X_scaled, y_pos_targets[pos])
            trained_pos_models[pos] = model

        app.logger.info(f"--- Melatih Meta-Model Multi-Label untuk CB/AM ---")
        base_estimator = LogisticRegression(random_state=42, max_iter=200, solver='liblinear')
        meta_model = OneVsRestClassifier(base_estimator)
        meta_model.fit(meta_features, y_cb_targets)
        
        model_filename = f"{pasaran}_model_v{MODEL_VERSION}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)

        model_payload = {
            'model_version': MODEL_VERSION,
            'models': {'pos_models': trained_pos_models, 'meta_model': meta_model}, 
            'scaler': scaler, 'pasaran': pasaran
        }
        joblib.dump(model_payload, model_path)
        
        app.logger.info(f"Model berhasil disimpan di: {model_path}")
        return jsonify({ "status": "success", "message": f"Arsitektur Meta-Model (v{MODEL_VERSION}) untuk {pasaran} berhasil dilatih dan disimpan." })
    except Exception as e:
        app.logger.error(f"ERROR di /train: {e}", exc_info=True)
        return jsonify({"error": f"Gagal melatih model: {e}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        pasaran, tanggal_str = data['pasaran'], data['tanggal']
        
        model_path = find_latest_model(pasaran)
        if not model_path: return jsonify({"error": f"Model untuk {pasaran} tidak ditemukan."}), 404
        
        payload = joblib.load(model_path)
        
        df_full, error, _ = get_data(pasaran)
        if error: return jsonify({"error": error}), 500
        
        target_date = pd.to_datetime(tanggal_str)
        df_historis = df_full[df_full['date'] < target_date].copy()
        if len(df_historis) < MIN_HISTORICAL_DATA: return jsonify({"error": f"Data historis tidak cukup untuk prediksi (min {MIN_HISTORICAL_DATA} hari)."}), 400
        
        result = get_full_prediction(df_historis, payload['models'], payload['scaler'], target_date, pasaran)
        
        history_entry = PredictionHistory.query.filter_by(tanggal=tanggal_str, pasaran=pasaran).first()
        if not history_entry:
            history_entry = PredictionHistory(tanggal=tanggal_str, pasaran=pasaran)
            db.session.add(history_entry)
        history_entry.prediksi_cb, history_entry.am = result['cb'], ','.join(map(str, result['am']))
        history_entry.prediksi_as, history_entry.prediksi_kop = ','.join(map(str, result['posisi']['as'])), ','.join(map(str, result['posisi']['kop']))
        history_entry.prediksi_kepala, history_entry.prediksi_ekor = ','.join(map(str, result['posisi']['kepala'])), ','.join(map(str, result['posisi']['ekor']))
        actual_row = df_full[df_full['date'] == target_date]
        if not actual_row.empty:
            hasil_aktual_str = str(int(actual_row.iloc[0]['result'])).zfill(4)
            cb_status_bool = result['cb'] in {int(d) for d in hasil_aktual_str}
            history_entry.hasil, history_entry.status, history_entry.cb_status = hasil_aktual_str, 'CB' if cb_status_bool else 'Gagal', 'Ya' if cb_status_bool else 'Tidak'
        db.session.commit()
        
        if 'pos_probas' in result: del result['pos_probas']
        result['tanggal'] = target_date.strftime('%Y-%m-%d')
        return jsonify(result)
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"ERROR di /predict: {e}", exc_info=True)
        return jsonify({"error": f"Terjadi kesalahan internal: {e}"}), 500

# ... (Sisa kode endpoint lainnya tidak perlu diubah, disertakan untuk kelengkapan)

def run_evaluation_in_background(job_id, pasaran, tgl_awal_str, tgl_akhir_str):
    try:
        with shared_jobs_lock: evaluation_jobs[job_id] = {'status': 'running', 'progress': 0, 'message': 'Memulai...'}
        df, error, reloaded = get_data(pasaran)
        if error: raise RuntimeError(f"Gagal memuat data: {error}")
        
        model_path = find_latest_model(pasaran)
        if not model_path: raise FileNotFoundError(f"Model untuk {pasaran} tidak ditemukan.")
        payload = joblib.load(model_path)
        
        tgl_awal, tgl_akhir = pd.to_datetime(tgl_awal_str), pd.to_datetime(tgl_akhir_str)
        eval_dates = df[(df['date'] >= tgl_awal) & (df['date'] <= tgl_akhir)]
        if eval_dates.empty: raise ValueError("Tidak ada data untuk dievaluasi pada rentang tanggal.")

        daily_details, confusion_matrix = [], np.zeros((10, 10), dtype=int)
        for i, (idx, row) in enumerate(eval_dates.iterrows()):
            with shared_jobs_lock: evaluation_jobs[job_id].update({'progress': 10 + int((i/len(eval_dates))*85), 'message': f"Memprediksi {row['date'].strftime('%Y-%m-%d')}"})
            df_historis = df.iloc[:df.index.get_loc(idx)]
            if len(df_historis) < MIN_HISTORICAL_DATA: continue
            
            prediction = get_full_prediction(df_historis, payload['models'], payload['scaler'], row['date'], pasaran)
            actual_str, actual_digits = str(int(row['result'])).zfill(4), {int(d) for d in str(int(row['result'])).zfill(4)}
            for ad in actual_digits: confusion_matrix[ad][prediction['cb']] += 1
            
            pos_names = {'as': 'A', 'kop': 'C', 'kepala': 'K', 'ekor': 'E'}
            status_list = [f"Hit {pos_names[p]}" for i, p in enumerate(pos_names) if int(actual_str[i]) in prediction['posisi'][p]]
            daily_details.append({"tanggal": row['date'].strftime('%Y-%m-%d'), "hasil": actual_str, "pred_cb": prediction['cb'],
                                  "cb_status": "Hit" if prediction['cb'] in actual_digits else "Miss", "pred_am": prediction['am'],
                                  "pos_kandidat": ' '.join([f"{pos_names[p]}:{','.join(map(str,prediction['posisi'][p]))}" for p in pos_names]),
                                  "final_status": ', '.join(status_list) if status_list else "Miss"})
        
        summary = { k: {'hit': 0} for k in ['cb', 'am', 'as', 'kop', 'kepala', 'ekor'] }
        total = len(daily_details)
        for d in daily_details:
            if d['cb_status'] == 'Hit': summary['cb']['hit'] += 1
            if set(d['pred_am']) & {int(c) for c in d['hasil']}: summary['am']['hit'] += 1
            if 'Hit A' in d['final_status']: summary['as']['hit'] += 1
            if 'Hit C' in d['final_status']: summary['kop']['hit'] += 1
            if 'Hit K' in d['final_status']: summary['kepala']['hit'] += 1
            if 'Hit E' in d['final_status']: summary['ekor']['hit'] += 1
        for k in summary:
            summary[k]['miss'] = total - summary[k]['hit']
            summary[k]['accuracy'] = (summary[k]['hit'] / total * 100) if total > 0 else 0

        with shared_jobs_lock: evaluation_jobs[job_id].update({'status': 'complete', 'progress': 100, 'message': 'Evaluasi Selesai!', 'summary': summary, 'daily_details': daily_details, 'confusion_matrix': confusion_matrix.tolist()})
    except Exception as e:
        with shared_jobs_lock: evaluation_jobs[job_id].update({'status': 'error', 'message': str(e)})

def adjust_weights_task(job_id, pasaran, days_to_evaluate=30):
    try:
        with shared_jobs_lock: weight_adjustment_jobs[job_id] = {'status': 'running', 'progress': 0, 'message': f'Memulai...'}
        df, error, reloaded = get_data(pasaran)
        if error: raise RuntimeError(f"Gagal memuat data: {error}")

        model_path = find_latest_model(pasaran)
        if not model_path: raise FileNotFoundError(f"Model {pasaran} tidak ditemukan.")
        payload = joblib.load(model_path)
        
        end_date = df['date'].max()
        eval_df = df[df['date'] >= end_date - timedelta(days=days_to_evaluate)]
        if len(eval_df) < 10: raise ValueError(f"Data tidak cukup (<10 hari).")
        
        performance = {pos: {'ai_wins': 0, 'freq_wins': 0} for pos in ['as', 'kop', 'kepala', 'ekor']}
        for i, (_, row) in enumerate(eval_df.iterrows()):
            with shared_jobs_lock: weight_adjustment_jobs[job_id].update({'progress': 10 + int((i/len(eval_df))*80), 'message': f"Menganalisis hari ke-{i+1}"})
            df_historis = df[df['date'] < row['date']]
            if len(df_historis) < MIN_HISTORICAL_DATA: continue
            
            prediction = get_full_prediction(df_historis, payload['models'], payload['scaler'], row['date'], pasaran)
            for pos in performance:
                if "pos_probas" in prediction and np.argmax(prediction["pos_probas"][pos]) == row[pos]:
                    performance[pos]['ai_wins'] += 1
                if df_historis[pos].value_counts().idxmax() == row[pos]:
                    performance[pos]['freq_wins'] += 1
        
        updated = False
        with shared_weights_lock: 
            current_weights = posisi_weights.get(pasaran, {})
            for pos in performance:
                total = performance[pos]['ai_wins'] + performance[pos]['freq_wins']
                if total > 0:
                    new_ai = round(np.clip(performance[pos]['ai_wins']/total, 0.2, 0.8), 2)
                    if current_weights.get(pos, {}).get('ai') != new_ai:
                        current_weights[pos] = {'ai': new_ai, 'freq': round(1.0 - new_ai, 2)}
                        updated = True
            if updated: posisi_weights[pasaran] = current_weights

        if updated:
            with open(WEIGHTS_CONFIG_PATH, 'w') as f: json.dump(posisi_weights, f, indent=2)
            msg = f'Sukses! Bobot untuk {pasaran.upper()} telah diperbarui.'
        else: msg = 'Selesai. Tidak ada perubahan bobot yang signifikan.'
        
        with shared_jobs_lock: weight_adjustment_jobs[job_id].update({'status': 'complete', 'progress': 100, 'message': msg})
    except Exception as e:
        with shared_jobs_lock: weight_adjustment_jobs[job_id].update({'status': 'error', 'message': str(e)})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    job_id = str(uuid.uuid4())
    data = request.json
    thread = threading.Thread(target=run_evaluation_in_background, args=(job_id, data['pasaran'], data['tgl_awal'], data['tgl_akhir']))
    thread.daemon = True; thread.start()
    return jsonify({"status": "started", "job_id": job_id})

@app.route('/update-weights', methods=['POST'])
def update_weights():
    job_id = str(uuid.uuid4())
    pasaran = request.json.get('pasaran')
    thread = threading.Thread(target=adjust_weights_task, args=(job_id, pasaran))
    thread.daemon = True; thread.start()
    return jsonify({"status": "started", "job_id": job_id})

@app.route('/evaluate-status/<job_id>', methods=['GET'])
def evaluate_status(job_id):
    with shared_jobs_lock: job = evaluation_jobs.get(job_id)
    return jsonify(job if job else {"status": "not_found"})

@app.route('/update-weights-status/<job_id>', methods=['GET'])
def update_weights_status(job_id):
    with shared_jobs_lock: job = weight_adjustment_jobs.get(job_id)
    return jsonify(job if job else {"status": "not_found"})

@app.route('/get-last-update', methods=['GET'])
def get_last_update():
    try:
        pasaran = request.args.get('pasaran')
        if not pasaran: return jsonify({"error": "Parameter 'pasaran' tidak ditemukan."}), 400
        df, error, _ = get_data(pasaran)
        if error: return jsonify({"error": error}), 500
        if df is None or df.empty: return jsonify({"error": f"Tidak ada data valid untuk pasaran {pasaran}."}), 404
        last_update = df['date'].max().strftime('%Y-%m-%d')
        return jsonify({"last_update": last_update})
    except Exception as e:
        app.logger.error(f"Error tak terduga di /get-last-update: {e}", exc_info=True)
        return jsonify({"error": "Terjadi kesalahan internal."}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        db.session.query(PredictionHistory).delete()
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Gagal menghapus riwayat."}), 500

@app.route('/refresh-data', methods=['POST'])
def refresh_data():
    pasaran = request.json.get('pasaran')
    _, error, reloaded = get_data(pasaran, force_reload=True)
    if error: return jsonify({"error": error}), 500
    if reloaded:
        keys_to_del = [k for k in feature_cache if k.startswith(pasaran)]
        for k in keys_to_del: del feature_cache[k]
        app.logger.info(f"Data {pasaran.upper()} diperbarui, cache fitur dihapus.")
    return jsonify({"status": "success", "message": f"Data untuk {pasaran} berhasil dimuat ulang."})

def shutdown_handler(signum, frame):
    sys.exit(0)

if __name__ == '__main__':
    load_weights()
    from waitress import serve
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    app.logger.info("Server Waitress dimulai pada http://0.0.0.0:8080")
    serve(app, host='0.0.0.0', port=8080)