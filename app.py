import os
import io
import time
import joblib
import requests
import numpy as np
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
import math
from scipy.stats import linregress
from itertools import combinations
import signal
import sys

MODEL_VERSION = "2.0"

# =================================================================
# ## KONFIGURASI APLIKASI, LOGGING, & DATABASE
# =================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# KOREKSI FINAL: Secara eksplisit memberitahu Flask untuk mencari file HTML di direktori saat ini (root).
app = Flask(__name__, template_folder='.')

# Konfigurasi Direktori
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Konfigurasi Logging
log_file = os.path.join(LOGS_DIR, 'app.log')
handler = RotatingFileHandler(log_file, maxBytes=1048576, backupCount=3)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info("Aplikasi Dimulai.")

# Konfigurasi Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app_database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model Database
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.String(10), nullable=False)
    pasaran = db.Column(db.String(50), nullable=False)
    prediksi_cb = db.Column(db.Integer, nullable=False)
    am = db.Column(db.String(50), nullable=True)
    prediksi_kop = db.Column(db.String(50), nullable=True)
    prediksi_kepala = db.Column(db.String(50), nullable=True)
    prediksi_ekor = db.Column(db.String(50), nullable=True)
    hasil = db.Column(db.String(20), nullable=True, default='-')
    status = db.Column(db.String(20), nullable=True, default='Menunggu')
    cb_status = db.Column(db.String(10), nullable=True, default='...')
    __table_args__ = (db.UniqueConstraint('tanggal', 'pasaran', name='_tanggal_pasaran_uc'),)

with app.app_context():
    db.create_all()

# Konfigurasi Lainnya
DATA_URLS = {
    "china": "https://raw.githubusercontent.com/widaditulus/4D/main/china_data.csv",
    "hk": "https://raw.githubusercontent.com/widaditulus/4D/main/hk_data.csv",
    "magnum": "https://raw.githubusercontent.com/widaditulus/4D/main/magnum_data.csv",
    "sgp": "https://raw.githubusercontent.com/widaditulus/4D/main/sgp_data.csv",
    "sydney": "https://raw.githubusercontent.com/widaditulus/4D/main/sydney_data.csv",
    "taiwan": "https://raw.githubusercontent.com/widaditulus/4D/main/taiwan_data.csv",
}
data_cache = {}
MIN_HISTORICAL_DATA = 91

# =================================================================
# ## FUNGSI HELPER & PEMUATAN DATA
# =================================================================
def _get_remote_last_modified(url: str):
    try:
        response = requests.head(url, timeout=10)
        response.raise_for_status()
        last_modified_str = response.headers.get('Last-Modified')
        if last_modified_str:
            return parsedate_to_datetime(last_modified_str)
    except requests.exceptions.RequestException:
        return None

def _download_and_save(url: str, local_path: str):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        return True, None
    except requests.exceptions.RequestException as e:
        return False, f"Gagal mengunduh data dari {url}: {e}"

def get_data(pasaran: str, force_reload: bool = False):
    if pasaran in data_cache and not force_reload:
        return data_cache[pasaran].copy(), None
    url = DATA_URLS.get(pasaran)
    local_path = os.path.join(DATA_DIR, f"{pasaran}_data.csv")
    should_download = False
    if force_reload or not os.path.exists(local_path):
        should_download = True
    else:
        local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
        remote_mtime = _get_remote_last_modified(url)
        if remote_mtime and remote_mtime > local_mtime:
            should_download = True
    if should_download:
        success, error = _download_and_save(url, local_path)
        if not success:
            return (None, error) if not os.path.exists(local_path) else (pd.read_csv(local_path), None)
    try:
        df = pd.read_csv(local_path)
        df = df.dropna(subset=['date', 'result'])
        for col in ['as', 'kop', 'kepala', 'ekor']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['as', 'kop', 'kepala', 'ekor']).astype({'as': int, 'kop': int, 'kepala': int, 'ekor': int})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        data_cache[pasaran] = df.copy()
        return df, None
    except Exception as e:
        return None, f"Gagal memproses file data lokal '{local_path}': {e}"

# =================================================================
# ## FUNGSI FEATURE ENGINEERING
# =================================================================

def calculate_slope(y):
    x = np.arange(len(y))
    slope, _, _, _, _ = linregress(x, y)
    return slope if not np.isnan(slope) else 0

def create_features_and_targets_vectorized(df, pasaran):
    app.logger.info(f"Memulai pembuatan fitur lengkap untuk pasaran: {pasaran.upper()}...")
    
    all_digits_in_rows = df[['as', 'kop', 'kepala', 'ekor']].values
    y = np.array([[1 if d in row else 0 for d in range(10)] for row in all_digits_in_rows])
    
    historical_feature_dfs = []
    
    for i in range(10):
        df[f'digit_{i}'] = ((df['as'] == i) | (df['kop'] == i) | (df['kepala'] == i) | (df['ekor'] == i)).astype(int)
    frekuensi_total = df[[f'digit_{i}' for i in range(10)]].expanding(min_periods=1).sum().shift(1).fillna(0)
    for i in range(10): historical_feature_dfs.append(frekuensi_total[f'digit_{i}'].rename(f'frekuensi_{i}'))

    for pos in ['as', 'kop', 'kepala', 'ekor']:
        pos_dummies = pd.get_dummies(df[pos], prefix=f'frekuensi_{pos}').reindex(columns=[f'frekuensi_{pos}_{i}' for i in range(10)], fill_value=0)
        pos_freq = pos_dummies.expanding(min_periods=1).sum().shift(1).fillna(0)
        historical_feature_dfs.append(pos_freq)

    for p in [7, 30, 90]:
        for i in range(10):
            col = f'digit_{i}'
            rolling_window = df[col].rolling(window=p)
            historical_feature_dfs.append(rolling_window.mean().shift(1).fillna(0).rename(f'rolling_mean_{p}d_{i}'))
            historical_feature_dfs.append(rolling_window.std().shift(1).fillna(0).rename(f'rolling_std_{p}d_{i}'))
            historical_feature_dfs.append(rolling_window.apply(calculate_slope, raw=True).shift(1).fillna(0).rename(f'rolling_slope_{p}d_{i}'))
    
    df['row_num'] = np.arange(len(df))
    for i in range(10):
        seen_at = df['row_num'].where(df[f'digit_{i}'] == 1)
        last_seen = seen_at.ffill().fillna(-1)
        cold_col = df['row_num'] - last_seen
        historical_feature_dfs.append(cold_col.shift(1).fillna(len(df)).rename(f'cold_score_{i}'))

    digits_arr = df[['as', 'kop', 'kepala', 'ekor']].values
    df['ganjil_count'] = (digits_arr % 2 != 0).sum(axis=1)
    df['genap_count'] = (digits_arr % 2 == 0).sum(axis=1)
    df['kecil_count'] = (digits_arr < 5).sum(axis=1)
    df['besar_count'] = (digits_arr >= 5).sum(axis=1)
    for prop in ['ganjil', 'genap', 'kecil', 'besar']:
        ratio_hist = (df[f'{prop}_count'] / 4).expanding(min_periods=1).mean().shift(1).fillna(0.5)
        historical_feature_dfs.append(ratio_hist.rename(f'rasio_hist_{prop}'))

    all_possible_pairs = list(combinations(range(10), 2))
    pair_columns = {f'pair_{p[0]}{p[1]}': [] for p in all_possible_pairs}
    for row in all_digits_in_rows:
        unique_digits_in_row = set(row)
        row_pairs = set(combinations(sorted(list(unique_digits_in_row)), 2))
        for p_key, p_val in zip(pair_columns.keys(), all_possible_pairs):
            pair_columns[p_key].append(1 if p_val in row_pairs else 0)
    for p_key, p_data in pair_columns.items():
        s = pd.Series(p_data, name=p_key)
        freq_hist = s.expanding(min_periods=1).mean().shift(1).fillna(0)
        historical_feature_dfs.append(freq_hist)

    future_feature_dfs = []
    future_feature_dfs.append(pd.Series(np.sin(2 * np.pi * df['date'].dt.dayofweek / 7), name='day_of_week_sin'))
    future_feature_dfs.append(pd.Series(np.cos(2 * np.pi * df['date'].dt.dayofweek / 7), name='day_of_week_cos'))
    future_feature_dfs.append(pd.Series(np.sin(2 * np.pi * df['date'].dt.dayofyear / 366), name='day_of_year_sin'))
    future_feature_dfs.append(pd.Series(np.cos(2 * np.pi * df['date'].dt.dayofyear / 366), name='day_of_year_cos'))
    
    if pasaran == 'sgp':
        is_before_off_day = df['date'].dt.dayofweek.isin([0, 3]).astype(int)
        future_feature_dfs.append(is_before_off_day.rename('sgp_before_off_day'))
    else:
        future_feature_dfs.append(pd.Series(np.zeros(len(df)), name='sgp_before_off_day'))

    all_feature_dfs = historical_feature_dfs + future_feature_dfs
    X_df = pd.concat(all_feature_dfs, axis=1)
    
    X_df.dropna(inplace=True)
    valid_indices = X_df.index
    X = X_df.values
    y = y[valid_indices]
    
    app.logger.info(f"Pembuatan fitur lengkap selesai. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y, X_df

def create_dataset_for_nn(df, pasaran):
    X, y, _ = create_features_and_targets_vectorized(df.copy(), pasaran)
    if len(X) == 0:
        return None, None
    return X, y

def _get_future_only_features(target_date, pasaran):
    features = {}
    features['day_of_week_sin'] = np.sin(2 * np.pi * target_date.dayofweek / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * target_date.dayofweek / 7)
    features['day_of_year_sin'] = np.sin(2 * np.pi * target_date.dayofyear / 366)
    features['day_of_year_cos'] = np.cos(2 * np.pi * target_date.dayofyear / 366)
    
    if pasaran == 'sgp':
        features['sgp_before_off_day'] = 1 if target_date.dayofweek in [0, 3] else 0
    else:
        features['sgp_before_off_day'] = 0
    
    return pd.Series(features)

def get_full_prediction(df_historis, model, scaler, target_date, all_features_df, pasaran):
    last_features_row = all_features_df.iloc[-1:].copy()
    future_features = _get_future_only_features(target_date, pasaran)
    
    for col, value in future_features.items():
        last_features_row[col] = value
        
    input_vector = last_features_row.values.reshape(1, -1)
    
    if input_vector.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Error: Jumlah fitur input ({input_vector.shape[1]}) tidak cocok dengan scaler ({scaler.n_features_in_})")

    input_scaled = scaler.transform(input_vector)
    skor_ai_proba = model.predict_proba(input_scaled)
    skor_ai = [p[0][1] for p in skor_ai_proba]
    skor_final = sorted(enumerate(skor_ai), key=lambda x: x[1], reverse=True)
    am_candidates = [item[0] for item in skor_final[:4]]
    cb = skor_final[0][0]
    posisi = {}
    for pos in ['kop', 'kepala', 'ekor']:
        frekuensi_pos = df_historis[pos].value_counts().reindex(range(10), fill_value=0)
        total_data = len(df_historis)
        pos_scores = [(d, (0.6 * skor_ai[d]) + (0.4 * (frekuensi_pos[d] / total_data))) for d in range(10)]
        posisi[pos] = [item[0] for item in sorted(pos_scores, key=lambda x: x[1], reverse=True)[:3]]
    return {"cb": cb, "am": sorted(am_candidates), "posisi": posisi}


# =================================================================
# ## ENDPOINTS FLASK (API)
# =================================================================

@app.route('/')
def index(): 
    return render_template('index.html')

# KOREKSI FINAL: Mengarahkan pencarian file JS ke direktori saat ini (root) secara benar.
@app.route('/main_refactored.js')
def serve_js():
    return send_from_directory('.', 'main_refactored.js')

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        pasaran = data['pasaran']
        config = data.get('config', {})
        
        app.logger.info("==========================================================")
        app.logger.info(f"PROSES PELATIHAN DIMULAI UNTUK PASARAN: {pasaran.upper()}")
        app.logger.info(f"Konfigurasi Hyperparameter: {config}")
        app.logger.info("----------------------------------------------------------")

        app.logger.info("Langkah 1: Memuat dan memproses data historis...")
        df, error = get_data(pasaran, force_reload=True)
        if error: return jsonify({"error": error}), 500
        app.logger.info(f"Data berhasil dimuat. Total {len(df)} baris data ditemukan.")

        app.logger.info("Langkah 2: Melakukan rekayasa fitur (feature engineering)...")
        X, y, all_features_df = create_features_and_targets_vectorized(df.copy(), pasaran)
        
        if X is None: return jsonify({"error": f"Data historis tidak cukup (butuh min {MIN_HISTORICAL_DATA} hari)."}), 400
        app.logger.info(f"Rekayasa fitur selesai. Ukuran matriks fitur (X): {X.shape[0]} baris, {X.shape[1]} fitur.")

        app.logger.info("Langkah 3: Melakukan penskalaan data (StandardScaler)...")
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        app.logger.info("Penskalaan data selesai.")

        app.logger.info("Langkah 4: Menginisialisasi model Neural Network (MLPClassifier)...")
        model = MLPClassifier(
            hidden_layer_sizes=(config.get('h1', 64), config.get('h2', 32)),
            activation='relu', solver='adam', 
            learning_rate_init=config.get('learning_rate', 0.001), 
            max_iter=config.get('epochs', 150),
            random_state=42, early_stopping=True, 
            n_iter_no_change=config.get('patience', 15),
            verbose=True
        )
        
        app.logger.info("Langkah 5: Memulai proses FIT (pelatihan model)...")
        multi_target_model = MultiOutputClassifier(model, n_jobs=-1).fit(X_scaled, y)
        app.logger.info("...Proses FIT selesai.")
        
        app.logger.info("Langkah 6: Menyimpan model, scaler, dan data fitur ke file...")
        model_payload = {
            'model_version': MODEL_VERSION,
            'model': multi_target_model, 
            'scaler': scaler, 
            'features_df': all_features_df, 
            'pasaran': pasaran
        }
        model_path = os.path.join(MODELS_DIR, f"{pasaran}_model.joblib")
        joblib.dump(model_payload, model_path)
        
        app.logger.info(f"Model berhasil disimpan di: {model_path}")
        app.logger.info("PROSES PELATIHAN SELESAI")
        app.logger.info("==========================================================")
        
        return jsonify({ "status": "success", "message": f"Model baru (Versi {MODEL_VERSION}) untuk {pasaran} berhasil dilatih dan disimpan." })
    except Exception as e:
        app.logger.error(f"ERROR di /train: {e}", exc_info=True)
        return jsonify({"error": f"Gagal melatih model: {e}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        pasaran, tanggal_str = data['pasaran'], data['tanggal']
        model_path = os.path.join(MODELS_DIR, f"{pasaran}_model.joblib")
        if not os.path.exists(model_path): return jsonify({"error": f"Model untuk {pasaran} tidak ditemukan. Silakan latih model terlebih dahulu."}), 404
        
        payload = joblib.load(model_path)

        if payload.get('model_version') != MODEL_VERSION:
            return jsonify({"error": f"Model usang (versi {payload.get('model_version')}, dibutuhkan {MODEL_VERSION}). Harap latih ulang model untuk pasaran ini."}), 400

        model, scaler, all_features_df = payload['model'], payload['scaler'], payload.get('features_df')
        
        df_full, error = get_data(pasaran)
        if error: return jsonify({"error": error}), 500
        
        if all_features_df is None:
            return jsonify({"error": "Model tidak valid (tidak berisi features_df). Mohon latih ulang model."}), 500

        target_date = pd.to_datetime(tanggal_str)
        df_historis = df_full[df_full['date'] < target_date]

        if len(df_historis) < MIN_HISTORICAL_DATA: 
            return jsonify({"error": f"Data historis tidak cukup (butuh min {MIN_HISTORICAL_DATA} hari)."}), 400
        
        if df_historis.empty:
            return jsonify({"error": "Tidak ada data historis yang ditemukan sebelum tanggal yang diminta."}), 400
        
        last_historical_idx = df_historis.index[-1]
        
        if last_historical_idx not in all_features_df.index:
            return jsonify({"error": "Data historis tidak sinkron dengan model. Latih ulang model dengan data terbaru."}), 400

        features_for_prediction = all_features_df.loc[:last_historical_idx]
        
        result = get_full_prediction(df_historis, model, scaler, target_date, features_for_prediction, pasaran)
        
        history_entry = PredictionHistory.query.filter_by(tanggal=tanggal_str, pasaran=pasaran).first()
        if not history_entry:
            history_entry = PredictionHistory(tanggal=tanggal_str, pasaran=pasaran)
            db.session.add(history_entry)
        history_entry.prediksi_cb = result['cb']
        history_entry.am = ','.join(map(str, result['am']))
        history_entry.prediksi_kop = ','.join(map(str, result['posisi']['kop']))
        history_entry.prediksi_kepala = ','.join(map(str, result['posisi']['kepala']))
        history_entry.prediksi_ekor = ','.join(map(str, result['posisi']['ekor']))
        actual_row = df_full[df_full['date'] == target_date]
        if not actual_row.empty:
            hasil_aktual_str = str(int(actual_row.iloc[0]['result'])).zfill(4)
            actual_digits = {int(d) for d in hasil_aktual_str}
            cb_status_bool = result['cb'] in actual_digits
            history_entry.hasil = hasil_aktual_str
            history_entry.status = 'CB' if cb_status_bool else 'Gagal'
            history_entry.cb_status = 'Ya' if cb_status_bool else 'Tidak'
        db.session.commit()
        result['tanggal'] = target_date.strftime('%Y-%m-%d')
        return jsonify(result)
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"ERROR di /predict: {e}", exc_info=True)
        return jsonify({"error": f"Terjadi kesalahan internal server: {e}"}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        pasaran, tgl_awal_str, tgl_akhir_str = data.get('pasaran'), data.get('tgl_awal'), data.get('tgl_akhir')
        model_path = os.path.join(MODELS_DIR, f"{pasaran}_model.joblib")
        if not os.path.exists(model_path): return jsonify({"error": f"Model untuk {pasaran} tidak ditemukan."}), 404
        
        payload = joblib.load(model_path)

        if payload.get('model_version') != MODEL_VERSION:
            return jsonify({"error": f"Model usang (versi {payload.get('model_version')}, dibutuhkan {MODEL_VERSION}). Harap latih ulang model."}), 400
        
        model, scaler, all_features_df = payload['model'], payload['scaler'], payload.get('features_df')
        
        df_full, error = get_data(pasaran)
        if error: return jsonify({"error": error}), 500
        
        tgl_awal, tgl_akhir = pd.to_datetime(tgl_awal_str), pd.to_datetime(tgl_akhir_str)
        
        if all_features_df is None:
             _, _, all_features_df = create_features_and_targets_vectorized(df_full.copy(), pasaran)

        eval_mask = (df_full['date'] >= tgl_awal) & (df_full['date'] <= tgl_akhir)
        eval_indices = df_full.index[eval_mask]
        
        if len(eval_indices) == 0:
            return jsonify({"error": "Tidak ada data yang dapat dievaluasi pada rentang tanggal tersebut."}), 404
            
        daily_details = []
        confusion_matrix = np.zeros((10, 10), dtype=int)
        
        for idx in eval_indices:
            if idx not in all_features_df.index: continue
            
            current_date = df_full.loc[idx, 'date']
            df_historis = df_full.iloc[:idx]
            
            last_hist_idx = df_historis.index[-1] if not df_historis.empty else -1
            if last_hist_idx == -1 or last_hist_idx not in all_features_df.index: continue

            features_historis_df = all_features_df.loc[:last_hist_idx]
            if features_historis_df.empty: continue
            
            prediction = get_full_prediction(df_historis, model, scaler, current_date, features_historis_df, pasaran)
            
            actual_result_str = str(int(df_full.loc[idx, 'result'])).zfill(4)
            actual_digits = {int(d) for d in actual_result_str}
            cb_status = "Hit" if prediction['cb'] in actual_digits else "Miss"
            am_found = list(set(prediction['am']).intersection(actual_digits))
            posisi_status_kop = int(actual_result_str[1]) in prediction['posisi']['kop']
            posisi_status_kepala = int(actual_result_str[2]) in prediction['posisi']['kepala']
            posisi_status_ekor = int(actual_result_str[3]) in prediction['posisi']['ekor']
            posisi_status = "Hit" if any([posisi_status_kop, posisi_status_kepala, posisi_status_ekor]) else "Miss"
            pred_posisi_str = f"C:{','.join(map(str, prediction['posisi']['kop']))} K:{','.join(map(str, prediction['posisi']['kepala']))} E:{','.join(map(str, prediction['posisi']['ekor']))}"
            
            daily_details.append({
                "tanggal": current_date.strftime('%Y-%m-%d'), "hasil": actual_result_str, "pred_cb": prediction['cb'],
                "cb_status": cb_status, "pred_am": prediction['am'], "am_found": am_found,
                "pred_posisi": pred_posisi_str, "posisi_status": posisi_status,
            })
            for actual_digit in actual_digits:
                confusion_matrix[actual_digit][prediction['cb']] += 1
                
        if not daily_details:
             return jsonify({"error": "Tidak ada data yang dapat dievaluasi pada rentang tanggal ini."}), 404
             
        total_predictions = len(daily_details)
        cb_hits = sum(1 for d in daily_details if d['cb_status'] == 'Hit')
        am_hits = sum(1 for d in daily_details if len(d['am_found']) > 0)
        kop_hits = sum(1 for d in daily_details if int(d['hasil'][1]) in [int(n) for n in d['pred_posisi'].split(' ')[0].split(':')[1].split(',')])
        kepala_hits = sum(1 for d in daily_details if int(d['hasil'][2]) in [int(n) for n in d['pred_posisi'].split(' ')[1].split(':')[1].split(',')])
        ekor_hits = sum(1 for d in daily_details if int(d['hasil'][3]) in [int(n) for n in d['pred_posisi'].split(' ')[2].split(':')[1].split(',')])
        
        summary = {
            "cb": {"hit": cb_hits, "miss": total_predictions - cb_hits, "accuracy": cb_hits / total_predictions if total_predictions > 0 else 0},
            "am": {"hit": am_hits, "miss": total_predictions - am_hits, "accuracy": am_hits / total_predictions if total_predictions > 0 else 0},
            "kop": {"hit": kop_hits, "miss": total_predictions - kop_hits, "accuracy": kop_hits / total_predictions if total_predictions > 0 else 0},
            "kepala": {"hit": kepala_hits, "miss": total_predictions - kepala_hits, "accuracy": kepala_hits / total_predictions if total_predictions > 0 else 0},
            "ekor": {"hit": ekor_hits, "miss": total_predictions - ekor_hits, "accuracy": ekor_hits / total_predictions if total_predictions > 0 else 0},
        }
        return jsonify({"summary": summary, "daily_details": daily_details, "confusion_matrix": confusion_matrix.tolist()})
    except Exception as e:
        app.logger.error(f"ERROR di /evaluate: {e}", exc_info=True)
        return jsonify({"error": f"Terjadi kesalahan internal saat evaluasi: {e}"}), 500

@app.route('/get-history', methods=['GET'])
def get_history():
    try:
        records = PredictionHistory.query.order_by(PredictionHistory.tanggal.desc()).limit(50).all()
        return jsonify([{"tanggal": r.tanggal, "pasaran": r.pasaran, "prediksi_cb": r.prediksi_cb, "am": r.am, "hasil": r.hasil, "status": r.status} for r in records])
    except Exception as e:
        return jsonify({"error": "Gagal mengambil riwayat."}), 500

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
    _, error = get_data(pasaran, force_reload=True)
    if error: return jsonify({"error": error}), 500
    return jsonify({"status": "success"})

@app.route('/get-last-update', methods=['GET'])
def get_last_update():
    pasaran = request.args.get('pasaran')
    df, error = get_data(pasaran)
    if error: return jsonify({"error": error}), 500
    return jsonify({"last_update": df['date'].max().strftime('%Y-%m-%d')})

def shutdown_handler(signum, frame):
    print("\nSinyal keluar diterima. Membersihkan sumber daya...")
    app.logger.info("Sinyal keluar diterima. Membersihkan sumber daya...")
    
    print("Membersihkan cache data...")
    data_cache.clear()
    
    with app.app_context():
        print("Menutup koneksi database...")
        db.engine.dispose()
    
    print("Menghentikan logger...")
    logging.shutdown()
    
    print("Pembersihan selesai. Aplikasi berhenti.")
    sys.exit(0)

if __name__ == '__main__':
    from waitress import serve
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    app.logger.info("Server Waitress dimulai pada http://0.0.0.0:8080")
    serve(app, host='0.0.0.0', port=8080)