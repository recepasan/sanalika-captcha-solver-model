"""
Captcha yon siniflandirma - Flask API
=====================================
POST /solve
    {"clue": "<key>", "keys": ["<key>", "<key>", ...]}
    -> hangi key'in clue ile ayni yonu gosterdigini dondurur

POST /direction
    {"key": "<key>"}
    -> tek gorselin yonunu dondurur

GET  /health
    -> model yuklu mu, hangi ayarlarla

Onceki surume gore duzeltmeler
------------------------------
1) RENK SIRASI: onceki surum cv2.imread ile BGR okuyup modele veriyordu ama
   model RGB ile egitilmisti (example.py ceviriyordu, API cevirmiyordu).
   Artik goruntuler tf.image.decode_image ile dogrudan RGB olarak cozuluyor.
2) DISKE YAZMA YOK: onceki surum indirdigi captcha gorsellerini
   DOWNLOAD_FOLDER="dataset" icine yaziyordu, yani egitim veri klasorunu
   etiketlenmemis dosyalarla kirletiyordu. Artik indirilen veri bellekte
   islenir; hicbir dosya yazilmaz.
3) PATH TRAVERSAL: dosya adi kullanicidan gelen key ile uretiliyordu
   (f"{prefix}{key}{ext}"), "../" iceren bir key klasor disina yazabilirdi.
   Diske yazma kalktigi icin risk ortadan kalkti; ayrica key formati
   dogrulaniyor.
4) SKORLAMA: ham embedding mesafesi yerine yon uyumu olasiligi kullaniliyor,
   TTA uygulaniyor ve emin olunmayan sonuclar needs_retry ile isaretleniyor.

GUVENLIK - deploy etmeden once okuyun
-------------------------------------
Bu endpoint kimlik dogrulamasi ICERMEZ. Varsayilan olarak yalnizca
127.0.0.1'e baglanir, yani sadece ayni makineden erisilebilir.
  - Aga acmak icin  : --host 0.0.0.0  (bilinerek yapin)
  - Basit koruma icin: CAPTCHA_API_KEY ortam degiskenini ayarlayin; ayarliysa
    her istekte X-API-Key basligi zorunlu olur.
Aga acip API key ayarlamazsaniz, agdaki herkes servisi kullanabilir.

Kullanim:
    python example-with-api.py
    CAPTCHA_API_KEY=gizli python example-with-api.py --host 0.0.0.0
"""

import argparse
import concurrent.futures
import os
import re
import sys
import threading
import time
from urllib.parse import urljoin

import requests
from flask import Flask, jsonify, request

from direction_infer import DirectionClassifier

# ============================================
# AYARLAR
# ============================================
META_PATH = "direction_embedding_meta.json"
CAPTCHA_BASE_URL = "https://fs.sanalika.com/global/captcha/"
TTA_VIEWS = 4
MIN_MARGIN = 0.10
DOWNLOAD_TIMEOUT = 5
MAX_KEYS = 16
MAX_IMAGE_BYTES = 4 * 1024 * 1024          # 4 MB
# Key formati: captcha anahtarlari nanoid (21 karakter, alfabe A-Za-z0-9_-).
# '.' ve '/' alfabede olmadigi icin yol gezinme yapisal olarak imkansiz.
# Anahtar '_' veya '-' ile baslayabilir; ilk karakteri kisitlamak gercek
# anahtarlarin reddedilmesine yol aciyordu.
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
API_KEY = os.environ.get("CAPTCHA_API_KEY", "")

# TLS dogrulamasi: yalnizca ortam degiskeniyle kapatilabilir.
# Kapatmak ortadaki adam saldirilarina acik hale getirir, gecici teshis icin.
VERIFY_TLS = os.environ.get("CAPTCHA_INSECURE_TLS", "0") != "1"

_clf = None
_clf_lock = threading.Lock()

app = Flask(__name__)


# ============================================
# MODEL (tek kez yuklenir)
# ============================================
def get_classifier():
    global _clf
    if _clf is None:
        with _clf_lock:
            if _clf is None:
                base = (os.path.dirname(sys.executable)
                        if getattr(sys, "frozen", False)
                        else os.path.dirname(os.path.abspath(__file__)))
                meta = os.path.join(base, META_PATH)
                if not os.path.exists(meta):
                    raise FileNotFoundError(
                        "Metadata bulunamadi: %s (once model-egitim.py calistirin)"
                        % meta
                    )
                print("Model yukleniyor...")
                _clf = DirectionClassifier(meta, tta_views=TTA_VIEWS)
                print("Model hazir | esik=%.3f | TTA=%d"
                      % (_clf.conf_threshold, _clf.tta_views))
    return _clf


# ============================================
# YARDIMCILAR
# ============================================
def require_api_key():
    """API_KEY ayarliysa X-API-Key basligini zorunlu kilar."""
    if not API_KEY:
        return None
    if request.headers.get("X-API-Key", "") != API_KEY:
        return jsonify({"error": "gecersiz veya eksik X-API-Key"}), 401
    return None


def valid_key(key):
    if not isinstance(key, str) or ".." in key:
        return False
    return bool(KEY_PATTERN.match(key))


def fetch_image(key, session):
    """Captcha gorselini bellege indirir. Diske yazmaz."""
    try:
        url = urljoin(CAPTCHA_BASE_URL, key)
        resp = session.get(url, timeout=DOWNLOAD_TIMEOUT, verify=VERIFY_TLS,
                           stream=True)
        resp.raise_for_status()
        data = resp.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
        if len(data) > MAX_IMAGE_BYTES:
            return {"key": key, "error": "gorsel cok buyuk (>%d bayt)"
                    % MAX_IMAGE_BYTES}
        if not data:
            return {"key": key, "error": "bos yanit"}
        return {"key": key, "data": data, "bytes": len(data)}
    except Exception as exc:
        return {"key": key, "error": str(exc)}


def fetch_all(keys):
    """Tum key'leri paralel indirir; sirayi korur."""
    out = [None] * len(keys)
    with requests.Session() as session:
        workers = min(10, max(len(keys), 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_image, k, session): i
                       for i, k in enumerate(keys)}
            for fut in concurrent.futures.as_completed(futures):
                out[futures[fut]] = fut.result()
    return out


# ============================================
# ENDPOINT: /solve
# ============================================
@app.route("/solve", methods=["POST"])
@app.route("/download-photo", methods=["POST"])   # eski isim, geriye uyumluluk
def solve():
    t_start = time.perf_counter()

    denied = require_api_key()
    if denied:
        return denied

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON govdesi bulunamadi"}), 400

    clue = data.get("clue")
    keys = data.get("keys")

    if not valid_key(clue):
        return jsonify({"error": "gecersiz clue"}), 400
    if not isinstance(keys, list) or not keys:
        return jsonify({"error": "keys dizisi bulunamadi"}), 400
    if len(keys) > MAX_KEYS:
        return jsonify({"error": "en fazla %d key" % MAX_KEYS}), 400
    bad = [k for k in keys if not valid_key(k)]
    if bad:
        return jsonify({"error": "gecersiz key(ler)", "keys": bad[:5]}), 400

    try:
        clf = get_classifier()
    except Exception as exc:
        return jsonify({"error": "model yuklenemedi: %s" % exc}), 500

    t_dl = time.perf_counter()
    fetched = fetch_all([clue] + keys)
    t_dl = time.perf_counter() - t_dl

    clue_res, cand_res = fetched[0], fetched[1:]
    if "error" in clue_res:
        return jsonify({"error": "clue indirilemedi: %s" % clue_res["error"]}), 502

    ok = [r for r in cand_res if "data" in r]
    failed = [{"key": r["key"], "error": r["error"]} for r in cand_res if "error" in r]
    if not ok:
        return jsonify({"error": "hicbir key indirilemedi", "failed": failed}), 502

    t_inf = time.perf_counter()
    try:
        res = clf.solve_captcha(clue_res["data"], [r["data"] for r in ok],
                                min_margin=MIN_MARGIN)
    except Exception as exc:
        return jsonify({"error": "tahmin hatasi: %s" % exc}), 500
    t_inf = time.perf_counter() - t_inf

    # data (bayt) alanlarini yanittan cikar, key ile eslestir
    results = []
    for c in res["candidates"]:
        results.append({
            "key": ok[c["index"]]["key"],
            "score": round(c["score"], 4),
            "direction": c["direction"],
            "direction_confidence": round(c["direction_confidence"], 4),
            "distance": round(c["distance"], 4),
        })
    winner = results[0]
    total = time.perf_counter() - t_start

    print("clue=%s -> %s | kazanan=%s skor=%.3f margin=%.3f%s | %.0f ms"
          % (clue, res["clue"]["direction"], winner["key"], winner["score"],
             res["margin"], " RETRY" if res["needs_retry"] else "", 1000 * total))

    return jsonify({
        "success": True,
        "winner": winner,
        "results": results,
        "clue": {
            "key": clue,
            "direction": res["clue"]["direction"],
            "confidence": round(res["clue"]["confidence"], 4),
            "certain": res["clue"]["certain"],
        },
        "margin": round(res["margin"], 4),
        "needs_retry": res["needs_retry"],
        "failed_downloads": failed,
        "timing": {
            "total_ms": round(1000 * total, 1),
            "download_ms": round(1000 * t_dl, 1),
            "inference_ms": round(1000 * t_inf, 1),
        },
    }), 200


# ============================================
# ENDPOINT: /direction
# ============================================
@app.route("/direction", methods=["POST"])
def direction():
    denied = require_api_key()
    if denied:
        return denied

    data = request.get_json(silent=True) or {}
    key = data.get("key")
    if not valid_key(key):
        return jsonify({"error": "gecersiz key"}), 400

    try:
        clf = get_classifier()
    except Exception as exc:
        return jsonify({"error": "model yuklenemedi: %s" % exc}), 500

    got = fetch_all([key])[0]
    if "error" in got:
        return jsonify({"error": "indirilemedi: %s" % got["error"]}), 502

    cls, conf = clf.predict(got["data"])
    return jsonify({
        "success": True,
        "key": key,
        "direction": None if cls is None else clf.class_names[cls],
        "confidence": round(float(conf), 4),
        "certain": cls is not None,
    }), 200


# ============================================
# ENDPOINT: /health
# ============================================
@app.route("/health", methods=["GET"])
def health():
    loaded = _clf is not None
    return jsonify({
        "status": "ok",
        "model_loaded": loaded,
        "meta": META_PATH,
        "tta_views": TTA_VIEWS,
        "confidence_threshold": (round(_clf.conf_threshold, 4) if loaded else None),
        "auth_required": bool(API_KEY),
        "tls_verify": VERIFY_TLS,
    }), 200


# ============================================
# MAIN
# ============================================
def main():
    p = argparse.ArgumentParser(description="Captcha cozucu API")
    p.add_argument("--host", default="127.0.0.1",
                   help="varsayilan 127.0.0.1 (sadece yerel). Aga acmak icin 0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--preload", action="store_true",
                   help="modeli baslangicta yukle (ilk istek hizli olsun)")
    args = p.parse_args()

    print("=" * 62)
    print("CAPTCHA COZUCU API")
    print("=" * 62)
    print("  metadata     : %s" % META_PATH)
    print("  TTA views    : %d" % TTA_VIEWS)
    print("  adres        : http://%s:%d" % (args.host, args.port))
    print("  kimlik dogr. : %s" % ("X-API-Key zorunlu" if API_KEY
                                   else "YOK (CAPTCHA_API_KEY ayarlanmadi)"))
    print("  TLS dogrulama: %s" % ("acik" if VERIFY_TLS else "KAPALI (guvensiz)"))
    if args.host not in ("127.0.0.1", "localhost") and not API_KEY:
        print("\n  UYARI: servis aga acik ve kimlik dogrulamasi yok.")
        print("         CAPTCHA_API_KEY ortam degiskenini ayarlamaniz onerilir.")
    print("\nEndpointler:")
    print("  POST /solve      {\"clue\": \"...\", \"keys\": [\"...\"]}")
    print("  POST /direction  {\"key\": \"...\"}")
    print("  GET  /health")
    print("\nOrnek:")
    print("  curl -X POST http://%s:%d/solve \\" % (args.host, args.port))
    print("    -H 'Content-Type: application/json' \\")
    if API_KEY:
        print("    -H 'X-API-Key: <anahtar>' \\")
    print("    -d '{\"clue\": \"abc\", \"keys\": [\"k1\", \"k2\", \"k3\"]}'")
    print("=" * 62 + "\n")

    if args.preload:
        get_classifier()

    app.run(debug=False, host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
