"""
Yön Sınıflandırma - API Örneği
==============================
Flask API ile captcha çözme servisi.

Kullanım:
    python example-with-api.py

Endpoint:
    POST http://localhost:5000/download-photo
    Body: {"clue": "abc123", "keys": ["key1", "key2", "key3"]}
"""

from flask import Flask, jsonify, request
import numpy as np
import cv2
import tensorflow as tf
import os
import sys
import time
import concurrent.futures
from urllib.parse import urljoin
import requests
import threading
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================
# AYARLAR
# ============================================
MODEL_PATH = "direction_embedding_best_windows.keras"
IMG_SIZE = (128, 128)
CAPTCHA_BASE_URL = "https://fs.sanalika.com/global/captcha/"
DOWNLOAD_FOLDER = "dataset"

# Global model cache
_model_cache = None
_model_lock = threading.Lock()

app = Flask(__name__)


# ============================================
# MODEL
# ============================================
def get_model():
    """Model'i cache'leyerek sadece bir kez yükler"""
    global _model_cache

    if _model_cache is None:
        with _model_lock:
            if _model_cache is None:
                if getattr(sys, 'frozen', False):
                    application_path = os.path.dirname(sys.executable)
                else:
                    application_path = os.path.dirname(os.path.abspath(__file__))

                model_path = os.path.join(application_path, MODEL_PATH)

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model bulunamadı: {model_path}")

                print("🚀 Model yükleniyor (ilk kez)...")
                _model_cache = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        "lambda": lambda x: tf.math.l2_normalize(x, axis=1),
                        "Lambda": lambda x: tf.math.l2_normalize(x, axis=1)
                    },
                    safe_mode=False
                )
                print("✓ Model cache'lendi")

    return _model_cache


# ============================================
# ENDPOINT: /download-photo
# ============================================
@app.route('/download-photo', methods=['POST'])
def download_photo():
    """
    Captcha çözme endpoint'i.

    Request Body:
        {
            "clue": "abc123",
            "keys": ["key1", "key2", "key3", "key4"]
        }

    Response:
        {
            "success": true,
            "winner": {"key": "key2", "distance": 0.1234},
            "results": [...],
            "timing": {...}
        }
    """
    start_time = time.time()

    try:
        # JSON validasyonu
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON verisi bulunamadı"}), 400

        keys = data.get("keys")
        clue = data.get("clue")

        if not keys or not isinstance(keys, list):
            return jsonify({"error": "keys dizisi bulunamadı"}), 400

        if not clue:
            return jsonify({"error": "clue bulunamadı"}), 400

        # Paths
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))

        download_folder = os.path.join(application_path, DOWNLOAD_FOLDER)
        os.makedirs(download_folder, exist_ok=True)

        # ===================== PARALEL İNDİRME =====================
        download_start = time.time()
        print("📥 İndirme başlıyor...")

        def download_image(key, prefix=""):
            try:
                url = urljoin(CAPTCHA_BASE_URL, key)
                resp = requests.get(url, timeout=5, verify=False)
                resp.raise_for_status()
                ext = ".jpg" if "jpeg" in resp.headers.get("content-type", "").lower() else ".png"
                filename = f"{prefix}{key}{ext}"
                path = os.path.join(download_folder, filename)
                with open(path, "wb") as f:
                    f.write(resp.content)
                return {"key": key, "url": url, "filename": filename, "path": path}
            except Exception as e:
                print(f"✗ {key} indirilemedi: {str(e)}")
                return {"key": key, "error": str(e)}

        # Tüm indirmeleri paralel yap
        all_keys = [clue] + keys
        prefixes = ["clue_"] + [""] * len(keys)

        downloaded_files = []
        failed_downloads = []
        clue_info = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(download_image, k, p): (k, p) for k, p in zip(all_keys, prefixes)}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if "error" in result:
                    failed_downloads.append(result)
                else:
                    key, prefix = futures[future]
                    if prefix == "clue_":
                        clue_info = result
                        print(f"✓ Clue: {result['filename']}")
                    else:
                        downloaded_files.append(result)
                        print(f"✓ Key: {result['filename']}")

        download_time = time.time() - download_start
        print(f"⏱️  İndirme süresi: {download_time:.2f}s ({len(downloaded_files)} dosya)")

        if not clue_info or not downloaded_files:
            return jsonify({
                "error": "Clue veya key dosyası indirilemedi",
                "downloaded": len(downloaded_files),
                "failed": len(failed_downloads)
            }), 400

        # ===================== MODEL YÜKLEME =====================
        model_start = time.time()
        print("🧠 Model yükleniyor...")

        try:
            model = get_model()
        except Exception as e:
            return jsonify({"error": f"Model yüklenirken hata: {str(e)}"}), 500

        model_load_time = time.time() - model_start
        print(f"⏱️  Model yükleme: {model_load_time:.2f}s")

        # ===================== GÖRSEL İŞLEME =====================
        def preprocess_batch(img_paths):
            """Toplu ön işleme"""
            batch = []
            valid_indices = []
            for i, path in enumerate(img_paths):
                try:
                    img = cv2.imread(path)
                    if img is None:
                        print(f"✗ {path} okunamadı")
                        continue
                    img = cv2.resize(img, IMG_SIZE)
                    img = img.astype("float32") / 127.5 - 1.0
                    batch.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"✗ {path} hata: {str(e)}")
            return np.array(batch), valid_indices

        preprocess_start = time.time()
        print("🖼️  Görsel ön işleme başlıyor...")

        all_paths = [clue_info["path"]] + [f["path"] for f in downloaded_files]
        batch_images, valid_indices = preprocess_batch(all_paths)

        preprocess_time = time.time() - preprocess_start
        print(f"⏱️  Ön işleme süresi: {preprocess_time:.2f}s ({len(batch_images)} görsel)")

        if len(batch_images) == 0:
            return jsonify({"error": "Hiçbir görsel işlenemedi"}), 500

        # ===================== EMBEDDİNG =====================
        prediction_start = time.time()
        print("🔮 Embedding hesaplanıyor...")

        embeddings = model.predict(batch_images, verbose=0, batch_size=32)

        prediction_time = time.time() - prediction_start
        print(f"⏱️  Prediction süresi: {prediction_time:.2f}s")

        # ===================== MESAFE HESAPLAMA =====================
        clue_vec = embeddings[0]

        results = []
        key_start_idx = 1

        for i, f in enumerate(downloaded_files):
            embedding_idx = key_start_idx + i
            if embedding_idx < len(embeddings):
                dist = float(np.linalg.norm(clue_vec - embeddings[embedding_idx]))
                results.append({
                    "key": f["key"],
                    "distance": dist,
                    "path": f["path"]
                })
                print(f"{f['key']} → {dist:.4f}")

        if not results:
            return jsonify({"error": "Hiçbir mesafe hesaplanamadı"}), 500

        winner = min(results, key=lambda x: x["distance"])

        total_time = time.time() - start_time

        print(f"\n🏆 Kazanan: {winner['key']} (Mesafe: {winner['distance']:.4f})")
        print(f"⏱️  TOPLAM SÜRE: {total_time:.2f}s\n")

        return jsonify({
            "success": True,
            "clue": clue_info,
            "download_folder": download_folder,
            "results": sorted(results, key=lambda x: x["distance"]),
            "winner": winner,
            "failed_downloads": failed_downloads,
            "timing": {
                "total": round(total_time, 2),
                "download": round(download_time, 2),
                "model_load": round(model_load_time, 2),
                "preprocessing": round(preprocess_time, 2),
                "prediction": round(prediction_time, 2)
            }
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("="*60)
    print("CAPTCHA ÇÖZÜCÜ API")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Download klasörü: {DOWNLOAD_FOLDER}")
    print("\nEndpoint'ler:")
    print("  POST /download-photo - Captcha çöz")
    print("\nÖrnek istek:")
    print('  curl -X POST http://localhost:5000/download-photo \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"clue": "abc123", "keys": ["key1", "key2", "key3"]}\'')
    print("="*60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000)
