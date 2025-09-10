import os
import io
import time
import joblib
import requests
import numpy as np
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
import math
from scipy.stats import linregress

# =================================================================
# ## KONFIGURASI APLIKASI, LOGGING, & DATABASE
# =================================================================
app = Flask(__name__)

# Konfigurasi Direktori
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
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
# ## FUNGSI HELPER & REKAYASA FITUR
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
        
        for i in range(10):
            df[f'digit_{i}'] = df[['as', 'kop', 'kepala', 'ekor']].apply(lambda row: 1 if i in row.values else 0, axis=1)

        data_cache[pasaran] = df.copy()
        return df, None
    except Exception as e:
        return None, f"Gagal memproses file data lokal '{local_path}': {e}"

# --- Kumpulan Fungsi Feature Engineering ---
def calculate_calendar_features(target_date):
    return {
        'day_of_week_sin': np.sin(2 * np.pi * target_date.dayofweek / 7),
        'day_of_week_cos': np.cos(2 * np.pi * target_date.dayofweek / 7),
        'day_of_year_sin': np.sin(2 * np.pi * target_date.dayofyear / 366),
        'day_of_year_cos': np.cos(2 * np.pi * target_date.dayofyear / 366),
    }

def calculate_rolling_stats(df_window):
    stats = {}
    periods = [7, 30, 90]
    for digit in range(10):
        col = f'digit_{digit}'
        for p in periods:
            if len(df_window) >= p:
                rolling_window = df_window[col].rolling(window=p)
                stats[f'rolling_mean_{p}d_{digit}'] = rolling_window.mean().iloc[-1]
                stats[f'rolling_std_{p}d_{digit}'] = rolling_window.std().iloc[-1]
                series = df_window[col].tail(p).values
                slope, _, _, _, _ = linregress(np.arange(len(series)), series)
                stats[f'rolling_slope_{p}d_{digit}'] = slope if not np.isnan(slope) else 0
            else:
                stats.update({f'rolling_mean_{p}d_{digit}': 0, f'rolling_std_{p}d_{digit}': 0, f'rolling_slope_{p}d_{digit}': 0})
    return stats

def calculate_co_occurrence(df_window):
    co_occurrence_matrix = np.zeros((10, 10))
    for _, row in df_window.iterrows():
        digits = np.unique(row[['as', 'kop', 'kepala', 'ekor']].values)
        for i in range(len(digits)):
            for j in range(i, len(digits)):
                d1, d2 = digits[i], digits[j]
                co_occurrence_matrix[d1, d2] += 1
                if d1 != d2: co_occurrence_matrix[d2, d1] += 1
    return co_occurrence_matrix.flatten().tolist()

def calculate_number_properties(df_window):
    all_digits = df_window[['as', 'kop', 'kepala', 'ekor']].values.flatten()
    total = len(all_digits)
    return {
        'ganjil_ratio': np.sum(all_digits % 2 != 0) / total if total > 0 else 0,
        'genap_ratio': np.sum(all_digits % 2 == 0) / total if total > 0 else 0,
        'kecil_ratio': np.sum(all_digits < 5) / total if total > 0 else 0,
        'besar_ratio': np.sum(all_digits >= 5) / total if total > 0 else 0,
    }

def extract_comprehensive_features(df_window, target_date):
    if df_window.empty: return {}
    features = {}
    all_digits = df_window[['as', 'kop', 'kepala', 'ekor']].values
    
    features['frekuensi'] = np.bincount(all_digits.flatten(), minlength=10).tolist()
    for pos in ['as', 'kop', 'kepala', 'ekor']:
        features[f'frekuensi_{pos}'] = np.bincount(df_window[pos], minlength=10).tolist()
    last_seen = {d: (len(df_window) - 1 - df_window.index[df_window[f'digit_{d}'] == 1][-1]) if f'digit_{d}' in df_window and (df_window[f'digit_{d}'] == 1).any() else len(df_window) for d in range(10)}
    features['cold_score'] = list(last_seen.values())
    
    features.update(calculate_calendar_features(target_date))
    features.update(calculate_rolling_stats(df_window))
    features.update(calculate_number_properties(df_window))
    features['co_occurrence'] = calculate_co_occurrence(df_window)
    
    return features

def create_model_input(features):
    input_vector = []
    for d in range(10):
        input_vector.extend([
            features['frekuensi'][d], features['cold_score'][d],
            features['frekuensi_as'][d], features['frekuensi_kop'][d],
            features['frekuensi_kepala'][d], features['frekuensi_ekor'][d],
            features.get(f'rolling_mean_7d_{d}', 0), features.get(f'rolling_std_7d_{d}', 0), features.get(f'rolling_slope_7d_{d}', 0),
            features.get(f'rolling_mean_30d_{d}', 0), features.get(f'rolling_std_30d_{d}', 0), features.get(f'rolling_slope_30d_{d}', 0),
            features.get(f'rolling_mean_90d_{d}', 0), features.get(f'rolling_std_90d_{d}', 0), features.get(f'rolling_slope_90d_{d}', 0),
        ])
    input_vector.extend([
        features['day_of_week_sin'], features['day_of_week_cos'],
        features['day_of_year_sin'], features['day_of_year_cos'],
        features['ganjil_ratio'], features['genap_ratio'],
        features['kecil_ratio'], features['besar_ratio'],
    ])
    input_vector.extend(features['co_occurrence'])
    return np.array(input_vector).reshape(1, -1)

# UPDATED: Penambahan logging di fungsi ini
def create_dataset_for_nn(df):
    if len(df) < MIN_HISTORICAL_DATA: return None, None
    X_list, y_list = [], []
    total_rows = len(df)
    
    app.logger.info(f"Memulai pembuatan dataset dari {total_rows} baris data...")
    for i in range(MIN_HISTORICAL_DATA, total_rows):
        # Log progress setiap 50 baris
        if (i - MIN_HISTORICAL_DATA) % 50 == 0:
            app.logger.info(f"Memproses baris ke-{i} dari {total_rows}...")
            
        df_window = df.iloc[:i]
        current_row = df.iloc[i]
        features = extract_comprehensive_features(df_window, current_row['date'])
        input_vector = create_model_input(features)
        X_list.append(input_vector[0])
        actual_digits = set(current_row[['as', 'kop', 'kepala', 'ekor']].values)
        y_list.append([1 if d in actual_digits else 0 for d in range(10)])
        
    app.logger.info("Pembuatan dataset selesai.")
    return np.array(X_list), np.array(y_list)

def get_full_prediction(df_historis, model, scaler, target_date):
    features = extract_comprehensive_features(df_historis, target_date)
    input_vector = create_model_input(features)
    input_scaled = scaler.transform(input_vector)
    
    skor_ai_proba = model.predict_proba(input_scaled)
    skor_ai = [p[0][1] for p in skor_ai_proba]
    
    skor_final = sorted(enumerate(skor_ai), key=lambda x: x[1], reverse=True)
    am_candidates = [item[0] for item in skor_final[:4]]
    cb = skor_final[0][0]
    
    posisi = {}
    total_data = len(df_historis)
    for pos in ['kop', 'kepala', 'ekor']:
        pos_scores = [(d, (0.6 * skor_ai[d]) + (0.4 * (features[f'frekuensi_{pos}'][d] / total_data))) for d in range(10)]
        posisi[pos] = [item[0] for item in sorted(pos_scores, key=lambda x: x[1], reverse=True)[:3]]
        
    return {"cb": cb, "am": sorted(am_candidates), "posisi": posisi}

# =================================================================
# ## ENDPOINTS FLASK (API)
# =================================================================

@app.route('/')
def index(): return render_template('index.html')

# UPDATED: Penambahan logging di endpoint ini
@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        pasaran = data['pasaran']
        config = data.get('config', {})
        app.logger.info(f"Menerima permintaan pelatihan untuk pasaran: {pasaran.upper()}")
        
        df, error = get_data(pasaran, force_reload=True)
        if error: return jsonify({"error": error}), 500
        
        X, y = create_dataset_for_nn(df)
        if X is None: return jsonify({"error": f"Data historis tidak cukup (butuh min {MIN_HISTORICAL_DATA} hari)."}), 400
        app.logger.info(f"Dataset berhasil dibuat. Shape X: {X.shape}, Shape y: {y.shape}")
        
        app.logger.info("Memulai penskalaan data dengan StandardScaler...")
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        app.logger.info("Penskalaan data selesai.")
        
        model = MLPClassifier(hidden_layer_sizes=(config.get('h1', 32), config.get('h2', 16)), activation='relu', solver='adam', learning_rate_init=config.get('learning_rate', 0.001), max_iter=config.get('epochs', 100), random_state=42, early_stopping=True, n_iter_no_change=config.get('patience', 15))
        
        app.logger.info("Memulai pelatihan model MultiOutputClassifier dengan MLP...")
        multi_target_model = MultiOutputClassifier(model, n_jobs=-1).fit(X_scaled, y)
        app.logger.info("Pelatihan model selesai.")
        
        model_payload = {'model': multi_target_model, 'scaler': scaler}
        model_path = os.path.join(MODELS_DIR, f"{pasaran}_model.joblib")
        joblib.dump(model_payload, model_path)
        app.logger.info(f"Model berhasil disimpan di: {model_path}")
        
        return jsonify({ "status": "success", "message": f"Model baru untuk {pasaran} berhasil dilatih." })
    except Exception as e:
        app.logger.error(f"ERROR di /train: {e}", exc_info=True)
        return jsonify({"error": f"Gagal melatih model: {e}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        pasaran, tanggal_str = data['pasaran'], data['tanggal']
        model_path = os.path.join(MODELS_DIR, f"{pasaran}_model.joblib")
        if not os.path.exists(model_path): return jsonify({"error": f"Model untuk {pasaran} tidak ditemukan."}), 404
        
        payload = joblib.load(model_path)
        model, scaler = payload['model'], payload['scaler']
        df_full, error = get_data(pasaran)
        if error: return jsonify({"error": error}), 500
        
        target_date = pd.to_datetime(tanggal_str)
        df_historis = df_full[df_full['date'] < target_date]
        if len(df_historis) < MIN_HISTORICAL_DATA: return jsonify({"error": f"Data historis tidak cukup (butuh min {MIN_HISTORICAL_DATA} hari)."}), 400
        
        result = get_full_prediction(df_historis, model, scaler, target_date)
        
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
        model, scaler = payload['model'], payload['scaler']
        df_full, error = get_data(pasaran)
        if error: return jsonify({"error": error}), 500

        tgl_awal, tgl_akhir = pd.to_datetime(tgl_awal_str), pd.to_datetime(tgl_akhir_str)
        
        cb_hits, am_hits, kop_hits, kepala_hits, ekor_hits, total_predictions = 0, 0, 0, 0, 0, 0
        
        date_range = pd.date_range(start=tgl_awal, end=tgl_akhir)
        for current_date in date_range:
            actual_row = df_full[df_full['date'] == current_date]
            if actual_row.empty: continue

            df_historis = df_full[df_full['date'] < current_date]
            if len(df_historis) < MIN_HISTORICAL_DATA: continue
            
            total_predictions += 1
            prediction = get_full_prediction(df_historis, model, scaler, current_date)
            
            actual_result_str = str(int(actual_row.iloc[0]['result'])).zfill(4)
            actual_digits = {int(d) for d in actual_result_str}
            
            cb_status = "Hit" if prediction['cb'] in actual_digits else "Miss"
            if cb_status == "Hit": cb_hits += 1
            if len(set(prediction['am']).intersection(actual_digits)) > 0: am_hits += 1
            if int(actual_result_str[1]) in prediction['posisi']['kop']: kop_hits += 1
            if int(actual_result_str[2]) in prediction['posisi']['kepala']: kepala_hits += 1
            if int(actual_result_str[3]) in prediction['posisi']['ekor']: ekor_hits += 1

        if total_predictions == 0:
             return jsonify({"error": "Tidak ada data yang dapat dievaluasi pada rentang tanggal tersebut."}), 404

        summary = {
            "cb": {"hit": cb_hits, "miss": total_predictions - cb_hits, "accuracy": (cb_hits / total_predictions) if total_predictions > 0 else 0},
            "am": {"hit": am_hits, "miss": total_predictions - am_hits, "accuracy": (am_hits / total_predictions) if total_predictions > 0 else 0},
            "kop": {"hit": kop_hits, "miss": total_predictions - kop_hits, "accuracy": (kop_hits / total_predictions) if total_predictions > 0 else 0},
            "kepala": {"hit": kepala_hits, "miss": total_predictions - kepala_hits, "accuracy": (kepala_hits / total_predictions) if total_predictions > 0 else 0},
            "ekor": {"hit": ekor_hits, "miss": total_predictions - ekor_hits, "accuracy": (ekor_hits / total_predictions) if total_predictions > 0 else 0},
        }

        return jsonify({"summary": summary, "daily_details": []})
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

if __name__ == '__main__':
    from waitress import serve
    app.logger.info("Server Waitress dimulai pada http://0.0.0.0:8080")
    serve(app, host='0.0.0.0', port=8080)