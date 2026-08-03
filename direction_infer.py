"""
Captcha yon siniflandirma - cikarim modulu

model-egitim.py'nin urettigi dosyalari kullanir:
    direction_embedding.keras            embedding modeli (L2-normalize cikis)
    direction_embedding_centers.npy      sinif merkezleri
    direction_embedding_meta.json        esik + metadata

Uc kullanim
-----------
1) Captcha cozme (asil gorev): clue ile ayni yonu gosteren secenegi bul.
       clf.solve_captcha("clue.png", ["a.png", "b.png", "c.png"])
   Skorlama, embedding mesafesi yerine YON UYUMU olasiligi uzerinden yapilir:
       P(ayni yon) = sum_k p_clue[k] * p_secenek[k]
   Bu, clue belirsiz oldugunda tek bir yon kararina bagli kalmadigi icin
   ham mesafe karsilastirmasindan daha saglamdir.

2) Tek gorselden yon:
       cls, conf = clf.predict("gorsel.png")   # cls None ise emin degil

3) Embedding / mesafe (geriye donuk uyumluluk):
       clf.embeddings(["a.png", "b.png"])

TTA (test-time augmentation)
----------------------------
Goruntu cevrildiginde yon degisir, bu yuzden skorlar ortalanmadan once GERI
PERMUTE edilir:
    yatay cevirme : i -> (n - i) % n
    dikey cevirme : i -> (n/2 - i) % n
Bu permutasyon olmadan ortalama almak sonucu sessizce bozar.

CLI
---
    # Esigi TTA ile kalibre et ve meta.json'a yaz
    python direction_infer.py --meta direction_embedding_meta.json \
        --eval direction_embedding_predictions.csv --tta-views 4 --save-threshold

    # Uctan uca captcha cozme dogrulugu (val gorselleriyle sentetik gorevler)
    python direction_infer.py --meta direction_embedding_meta.json \
        --eval direction_embedding_predictions.csv --simulate 300 --options 4

    # Tek gorsel / tek captcha
    python direction_infer.py --meta ... --image gorsel.png
    python direction_infer.py --meta ... --solve clue.png a.png b.png c.png
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class DirectionClassifier:
    def __init__(self, meta_path, tta_views=2, conf_threshold=None):
        if tta_views not in (1, 2, 4):
            raise ValueError("tta_views 1, 2 veya 4 olmali")
        with open(meta_path) as fh:
            meta = json.load(fh)

        self.meta = meta
        self.model = tf.keras.models.load_model(meta["embedding_model"])
        self.centers = np.load(meta["centers"]).astype(np.float32)
        self.class_names = list(meta["class_names"])
        self.img_size = tuple(meta["img_size"])
        self.scale = float(meta.get("cosface_scale", 24.0))
        self.pair_threshold = float(meta.get("recommended_pair_threshold", 1.0))
        self.n = len(self.class_names)
        self.tta_views = int(tta_views)

        # Esik: once elle verilen, sonra TTA'ya ozel kalibrasyon, sonra genel
        thrs = meta.get("confidence_thresholds", {})
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        elif str(self.tta_views) in thrs:
            self.conf_threshold = float(thrs[str(self.tta_views)])
        else:
            self.conf_threshold = float(
                meta.get("recommended_confidence_threshold", 0.0)
            )

        n = self.n
        # Cevirme sonrasi sinif eslemeleri (involutif)
        self.perm_h = np.array([(n - i) % n for i in range(n)], dtype=np.int32)
        self.perm_v = np.array([(n // 2 - i) % n for i in range(n)], dtype=np.int32)
        self.perm_hv = self.perm_h[self.perm_v]

        # model.predict() cagri basina buyuk Python yuku tasir; tf.function ile
        # sarilmis dogrudan cagri degisken batch boyutunda tek grafik kullanir.
        h, w = self.img_size
        self._forward = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=[tf.TensorSpec([None, h, w, 3], tf.float32)],
            reduce_retracing=True,
        )

    # ------------------------------------------------------------------ girdi
    def preprocess(self, img):
        """Cikis: (H,W,3) float32 [-1,1].

        Kabul edilen girdiler:
            str / os.PathLike : dosya yolu
            bytes / bytearray : gorsel ICERIGI (indirilen veri, diske yazmadan)
            np.ndarray        : HxWx3 veya HxWx4 uint8 RGB

        DIKKAT: uint8 dizi verirken RGB olmali. cv2.imread BGR dondurur,
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ile cevirin - model RGB ile
        egitildi (tf.image.decode_image RGB uretir).
        """
        if isinstance(img, (bytes, bytearray)):
            x = tf.image.decode_image(bytes(img), channels=3, expand_animations=False)
        elif isinstance(img, (str, os.PathLike)):
            x = tf.image.decode_image(
                tf.io.read_file(str(img)), channels=3, expand_animations=False
            )
        else:
            arr = np.asarray(img)
            # Hizli yol: dogru boyutta uint8 geldiyse TF'e hic girmeden olcekle
            if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] >= 3 \
                    and arr.shape[:2] == self.img_size:
                return arr[..., :3].astype(np.float32) / 127.5 - 1.0
            x = tf.convert_to_tensor(arr)
            if x.shape[-1] == 4:
                x = x[..., :3]
        x = tf.image.resize(tf.cast(x, tf.float32), self.img_size, method="bilinear")
        return (x / 127.5 - 1.0).numpy()

    def preprocess_many(self, imgs):
        return np.stack([self.preprocess(i) for i in imgs])

    # ----------------------------------------------------------------- skorlar
    def scores_batch(self, imgs, return_embeddings=False):
        """imgs: on islenmis (B,H,W,3). Cikis: (B, n_classes) cos skorlari.

        TTA acikken cevrilmis gorunumlerin skorlari permutasyonla orijinal
        cerceveye tasinip ortalanir. Embedding dondurulurse yalnizca orijinal
        gorunumun embedding'i dondurulur (cevrilmis embedding permute edilemez).
        """
        views, perms = [imgs], [None]
        if self.tta_views >= 2:
            views.append(imgs[:, :, ::-1, :])      # yatay cevir
            perms.append(self.perm_h)
        if self.tta_views == 4:
            views.append(imgs[:, ::-1, :, :])      # dikey cevir
            perms.append(self.perm_v)
            views.append(imgs[:, ::-1, ::-1, :])   # ikisi birlikte
            perms.append(self.perm_hv)

        b = len(imgs)
        stacked = np.concatenate(views, axis=0).astype(np.float32)
        emb = self._forward(stacked).numpy()
        s = emb @ self.centers.T  # (views*B, n)

        total = np.zeros((b, self.n), dtype=np.float32)
        for k, perm in enumerate(perms):
            part = s[k * b : (k + 1) * b]
            total += part if perm is None else part[:, perm]
        total /= float(len(views))
        return (total, emb[:b]) if return_embeddings else total

    # -------------------------------------------------------- tek gorsel / yon
    def predict_batch(self, imgs, apply_threshold=True):
        """imgs: yol/dizi listesi. Cikis: [(cls|None, conf), ...]"""
        s = self.scores_batch(self.preprocess_many(imgs))
        out = []
        for row in s:
            cls = int(np.argmax(row))
            conf = float(row[cls])
            out.append((None, conf) if apply_threshold and conf < self.conf_threshold
                       else (cls, conf))
        return out

    def predict(self, img, apply_threshold=True):
        """Tek goruntu. Cikis: (cls|None, conf). cls None ise emin degil."""
        return self.predict_batch([img], apply_threshold)[0]

    def embeddings(self, imgs):
        """Geriye donuk uyumluluk: L2-normalize embedding matrisi (B, dim)."""
        _, emb = self.scores_batch(self.preprocess_many(imgs), return_embeddings=True)
        return emb

    # ------------------------------------------------------------ captcha cozme
    def solve_captcha(self, clue, candidates, min_margin=0.10):
        """clue ile ayni yonu gosteren secenegi bulur.

        Skorlama: P(ayni yon) = sum_k p_clue[k] * p_secenek[k]
        p, CosFace olcegiyle softmax'lanmis yon olasiligi.

        min_margin: en iyi ile ikinci arasindaki normalize fark bunun altinda
        kalirsa needs_retry True doner (yeni captcha istemek daha ucuz).
        """
        if not candidates:
            raise ValueError("candidates bos")

        imgs = [clue] + list(candidates)
        s, emb = self.scores_batch(self.preprocess_many(imgs), return_embeddings=True)
        p = softmax(self.scale * s, axis=1)

        agree = p[1:] @ p[0]                                  # (N,)
        dist = np.linalg.norm(emb[1:] - emb[0], axis=1)       # eski metrik
        total = float(agree.sum()) or 1.0
        norm_agree = agree / total

        order = np.argsort(-agree)
        best, second = int(order[0]), int(order[1]) if len(order) > 1 else None
        margin = float(norm_agree[best] - (norm_agree[second] if second is not None else 0.0))

        clue_cls = int(np.argmax(s[0]))
        clue_conf = float(s[0, clue_cls])

        cands = []
        for i in order:
            i = int(i)
            cls = int(np.argmax(s[i + 1]))
            cands.append({
                "index": i,
                "value": candidates[i],
                "agreement": float(agree[i]),
                "score": float(norm_agree[i]),
                "distance": float(dist[i]),
                "direction": self.class_names[cls],
                "direction_confidence": float(s[i + 1, cls]),
                "is_same_direction": bool(dist[i] < self.pair_threshold),
            })

        return {
            "winner": cands[0],
            "candidates": cands,
            "clue": {
                "value": clue,
                "direction": self.class_names[clue_cls],
                "confidence": clue_conf,
                "certain": bool(clue_conf >= self.conf_threshold),
            },
            "margin": margin,
            "needs_retry": bool(clue_conf < self.conf_threshold or margin < min_margin),
        }


# ---------------------------------------------------------------------------
# Kalibrasyon / degerlendirme
# ---------------------------------------------------------------------------
def gate_table(score, correct):
    print("    esik    kapsam   kapsanan_acc  reddedilen_dogru")
    for cov_target in (1.00, 0.98, 0.95, 0.92, 0.90, 0.85):
        thr = float(np.quantile(score, 1.0 - cov_target))
        mask = score >= thr
        acc = float(correct[mask].mean()) if mask.any() else 0.0
        print("    %5.3f  %6.2f%%   %7.2f%%      %4d"
              % (thr, 100.0 * mask.mean(), 100.0 * acc,
                 int((~mask & correct).sum())))


def read_val_csv(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError("%s bos" % csv_path)
    return [r["dosya"] for r in rows], [r["gercek"] for r in rows]


def evaluate(clf, csv_path, batch=64):
    paths, names = read_val_csv(csv_path)
    idx = {n: i for i, n in enumerate(clf.class_names)}
    y = np.array([idx[n] for n in names], dtype=np.int32)

    # Isinma: ilk cagri XLA derlemesi icerir, olcume dahil edilmez
    clf.scores_batch(clf.preprocess_many(paths[: min(batch, len(paths))]))

    scores, t_infer = [], 0.0
    for i in range(0, len(paths), batch):
        imgs = clf.preprocess_many(paths[i : i + batch])
        t0 = time.perf_counter()
        scores.append(clf.scores_batch(imgs))
        t_infer += time.perf_counter() - t0
    s = np.concatenate(scores, axis=0)

    pred = np.argmax(s, axis=1)
    conf = s[np.arange(len(pred)), pred]
    correct = pred == y

    print("\nTTA views: %d | gorsel: %d" % (clf.tta_views, len(y)))
    print("8-yon accuracy: %.2f%%  (%d hata)"
          % (100.0 * correct.mean(), int((~correct).sum())))
    print("inference: %.2f ms/gorsel (batch=%d, on isleme haric)"
          % (1000.0 * t_infer / len(paths), batch))
    print("\ncos tabanli red esigi:")
    gate_table(conf, correct)

    if (~correct).any():
        thr = float(conf[~correct].max()) + 0.005
        mask = conf >= thr
        print("\nTUM hatalari eleyen esik: %.3f -> kapsam %.2f%%, bedel %d dogru tahmin"
              % (thr, 100.0 * mask.mean(), int((~mask & correct).sum())))
        print("\nHatalar:")
        for i in np.where(~correct)[0]:
            print("  gercek=%-14s tahmin=%-14s conf=%.3f  %s"
                  % (clf.class_names[y[i]], clf.class_names[pred[i]], conf[i], paths[i]))
        return thr
    print("\nHata yok, esik gerekmiyor")
    return 0.0


def simulate(clf, csv_path, n_tasks=300, n_options=4, seed=42, min_margin=0.10):
    """Uctan uca captcha dogrulugu: val gorselleriyle sentetik gorevler kurar.

    Her gorevde clue'nun sinifindan 1 dogru secenek, diger siniflardan
    n_options-1 yanlis secenek bulunur. Egitim gorselleri kullanilmaz.
    """
    paths, names = read_val_csv(csv_path)
    by_class = defaultdict(list)
    for p, nm in zip(paths, names):
        by_class[nm].append(p)
    usable = sorted(c for c in by_class if len(by_class[c]) >= 2)
    if len(usable) < n_options:
        raise ValueError("yeterli sinif yok: %d sinif, %d secenek isteniyor"
                         % (len(usable), n_options))

    rng = np.random.default_rng(seed)
    hit = retry = hit_wo_retry = 0
    margins = []

    for _ in range(n_tasks):
        c = usable[int(rng.integers(len(usable)))]
        two = rng.choice(len(by_class[c]), 2, replace=False)
        clue, answer = by_class[c][two[0]], by_class[c][two[1]]

        others = [o for o in usable if o != c]
        picks = rng.choice(len(others), n_options - 1, replace=False)
        wrong = [by_class[others[i]][int(rng.integers(len(by_class[others[i]])))]
                 for i in picks]

        options = [answer] + wrong
        perm = rng.permutation(len(options))
        options = [options[i] for i in perm]
        answer_idx = int(np.where(perm == 0)[0][0])

        res = clf.solve_captcha(clue, options, min_margin=min_margin)
        ok = res["winner"]["index"] == answer_idx
        hit += int(ok)
        margins.append(res["margin"])
        if res["needs_retry"]:
            retry += 1
        else:
            hit_wo_retry += int(ok)

    answered = n_tasks - retry
    print("\n" + "=" * 62)
    print("UCTAN UCA CAPTCHA SIMULASYONU")
    print("=" * 62)
    print("gorev: %d | secenek/gorev: %d | TTA views: %d"
          % (n_tasks, n_options, clf.tta_views))
    print("cozum dogrulugu (filtresiz)      : %.2f%%  (%d/%d)"
          % (100.0 * hit / n_tasks, hit, n_tasks))
    if answered:
        print("cozum dogrulugu (retry filtresiz): %.2f%%  (%d/%d cevaplanan)"
              % (100.0 * hit_wo_retry / answered, hit_wo_retry, answered))
    print("yeniden deneme orani             : %.2f%%  (%d gorev)"
          % (100.0 * retry / n_tasks, retry))
    print("margin: ortalama=%.3f  min=%.3f" % (float(np.mean(margins)),
                                               float(np.min(margins))))
    print("rastgele tahmin taban cizgisi    : %.2f%%" % (100.0 / n_options))
    return hit / n_tasks


def benchmark(clf, csv_path, n_calls=200):
    """Bot dongusundeki gercek gecikme: tek goruntu, uctan uca."""
    paths, _ = read_val_csv(csv_path)
    paths = paths[: min(n_calls, len(paths))]
    raw = [(255.0 * (clf.preprocess(p) + 1.0) / 2.0).astype(np.uint8) for p in paths]

    for _ in range(5):
        clf.predict(raw[0])

    t0 = time.perf_counter()
    for arr in raw:
        clf.predict(arr)
    dt = time.perf_counter() - t0
    print("\nTek goruntu gecikmesi (tta_views=%d): %.2f ms/cagri  (%d cagri)"
          % (clf.tta_views, 1000.0 * dt / len(raw), len(raw)))
    print("  -> saniyede ~%.0f tahmin" % (len(raw) / dt))


def save_threshold(meta_path, tta_views, thr, headroom):
    """Kalibre edilen esigi meta.json'a yazar (TTA view sayisina gore)."""
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta.setdefault("confidence_thresholds", {})
    value = round(float(thr) + float(headroom), 3)
    meta["confidence_thresholds"][str(tta_views)] = value
    meta["tta_views"] = int(tta_views)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print("\n%s guncellendi: confidence_thresholds['%d'] = %.3f (headroom +%.3f)"
          % (meta_path, tta_views, value, headroom))


def main():
    p = argparse.ArgumentParser(description="Captcha yon cikarimi / kalibrasyon")
    p.add_argument("--meta", default="direction_embedding_meta.json")
    p.add_argument("--eval", help="<prefix>_predictions.csv (kalibrasyon icin)")
    p.add_argument("--simulate", type=int, metavar="N",
                   help="N sentetik captcha gorevi ile uctan uca dogruluk")
    p.add_argument("--options", type=int, default=4, help="simulasyonda secenek sayisi")
    p.add_argument("--image", help="tek goruntunun yonu")
    p.add_argument("--solve", nargs="+", metavar="YOL",
                   help="clue.png opt1.png opt2.png ... (ilki clue)")
    p.add_argument("--tta-views", type=int, default=4, choices=(1, 2, 4))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--min-margin", type=float, default=0.10)
    p.add_argument("--save-threshold", action="store_true",
                   help="kalibre edilen esigi meta.json'a yaz")
    p.add_argument("--headroom", type=float, default=0.01,
                   help="esige eklenecek guvenlik payi (val'e asiri uyuma karsi)")
    p.add_argument("--bench", action="store_true", help="tek goruntu gecikmesini olc")
    args = p.parse_args()

    clf = DirectionClassifier(args.meta, tta_views=args.tta_views)
    print("Model: %s | embedding: %d | esik: %.3f | TTA: %d"
          % (clf.meta["embedding_model"], clf.centers.shape[1],
             clf.conf_threshold, clf.tta_views))

    did = False
    if args.eval:
        thr = evaluate(clf, args.eval, args.batch)
        did = True
        if args.save_threshold:
            save_threshold(args.meta, args.tta_views, thr, args.headroom)
    if args.simulate:
        if not args.eval:
            p.error("--simulate icin --eval ile val CSV'si gerekiyor")
        simulate(clf, args.eval, args.simulate, args.options,
                 min_margin=args.min_margin)
        did = True
    if args.bench:
        if not args.eval:
            p.error("--bench icin --eval ile val CSV'si gerekiyor")
        benchmark(clf, args.eval)
        did = True
    if args.image:
        cls, conf = clf.predict(args.image)
        label = "EMIN DEGIL" if cls is None else clf.class_names[cls]
        print("\n%s -> %s (conf=%.3f)" % (args.image, label, conf))
        did = True
    if args.solve:
        if len(args.solve) < 2:
            p.error("--solve icin en az clue + 1 secenek gerekiyor")
        res = clf.solve_captcha(args.solve[0], args.solve[1:],
                                min_margin=args.min_margin)
        print("\nclue: %s -> %s (conf=%.3f%s)"
              % (res["clue"]["value"], res["clue"]["direction"],
                 res["clue"]["confidence"],
                 "" if res["clue"]["certain"] else ", EMIN DEGIL"))
        print("\nsecenekler (yon uyumuna gore sirali):")
        for i, c in enumerate(res["candidates"]):
            print("  %s%.3f  %-14s mesafe=%.3f  %s"
                  % ("-> " if i == 0 else "   ", c["score"], c["direction"],
                     c["distance"], c["value"]))
        print("\nkazanan: %s (margin=%.3f)%s"
              % (res["winner"]["value"], res["margin"],
                 "  [YENIDEN DENE]" if res["needs_retry"] else ""))
        did = True

    if not did:
        p.error("--eval, --simulate, --image, --solve veya --bench verin")


if __name__ == "__main__":
    main()
