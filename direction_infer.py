"""
Captcha yon siniflandirma - cikarim modulu (ONNX Runtime & Keras Destekli)

Kullanilan dosyalar:
    direction_embedding_v8.onnx          ONNX modeli (Onerilen: ~80 MB RAM, ultra hizli)
    direction_embedding_v8.keras         Keras modeli (Geriye donuk uyumluluk)
    direction_embedding_v8_centers.npy   sinif merkezleri
    direction_embedding_v8_meta.json     esik + metadata

Uc kullanim
-----------
1) Captcha cozme (asil gorev): clue ile ayni yonu gosteren secenegi bul.
       clf.solve_captcha("clue.png", ["a.png", "b.png", "c.png"])
   Skorlama, embedding mesafesi yerine YON UYUMU olasiligi uzerinden yapilir:
       P(ayni yon) = sum_k p_clue[k] * p_secenek[k]

2) Tek gorselden yon:
       cls, conf = clf.predict("gorsel.png")   # cls None ise emin degil

3) Embedding / mesafe:
       clf.embeddings(["a.png", "b.png"])
"""

import argparse
import csv
import io
import json
import os
import time
from collections import defaultdict

import numpy as np
from PIL import Image


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class DirectionClassifier:
    def __init__(self, meta_path, tta_views=2, conf_threshold=None, num_threads=2):
        if tta_views not in (1, 2, 4):
            raise ValueError("tta_views 1, 2 veya 4 olmali")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        base_dir = os.path.dirname(os.path.abspath(meta_path))

        def _resolve(rel):
            return rel if os.path.isabs(rel) else os.path.join(base_dir, rel)

        self.meta = meta
        model_path = _resolve(meta["embedding_model"])
        
        # ONNX veya Keras tespiti
        # Eger meta icinde .keras yaziyorsa ama ayni adda .onnx dosyasi varsa dogrudan ONNX tercih edilebilir
        onnx_alt = os.path.splitext(model_path)[0] + ".onnx"
        if model_path.endswith(".onnx") or os.path.exists(onnx_alt):
            actual_model = model_path if model_path.endswith(".onnx") else onnx_alt
            self.is_onnx = True
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError("ONNX modeli calistirmak icin 'onnxruntime' gerekli: pip install onnxruntime")
            
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = int(num_threads)
            sess_opts.inter_op_num_threads = int(num_threads)
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Mac CPU & CoreML destegi
            avail = ort.get_available_providers()
            providers = [p for p in ["CPUExecutionProvider"] if p in avail]
            self.session = ort.InferenceSession(actual_model, sess_options=sess_opts, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.model_file = actual_model
        else:
            self.is_onnx = False
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError("Keras modeli calistirmak icin 'tensorflow' gerekli: pip install tensorflow")
            
            self.model = tf.keras.models.load_model(model_path)
            self.model_file = model_path
            
            h, w = tuple(meta["img_size"])
            self._tf_forward = tf.function(
                lambda x: self.model(x, training=False),
                input_signature=[tf.TensorSpec([None, h, w, 3], tf.float32)],
                reduce_retracing=True,
            )

        self.centers = np.load(_resolve(meta["centers"])).astype(np.float32)
        self.class_names = list(meta["class_names"])
        self.img_size = tuple(meta["img_size"])
        self.scale = float(meta.get("cosface_scale", 24.0))
        self.pair_threshold = float(meta.get("recommended_pair_threshold", 1.0))
        self.n = len(self.class_names)
        self.tta_views = int(tta_views)

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
        self.perm_h = np.array([(n - i) % n for i in range(n)], dtype=np.int32)
        self.perm_v = np.array([(n // 2 - i) % n for i in range(n)], dtype=np.int32)
        self.perm_hv = self.perm_h[self.perm_v]

    def _forward(self, stacked):
        if self.is_onnx:
            return self.session.run(None, {self.input_name: stacked})[0]
        else:
            out = self._tf_forward(stacked)
            return out.numpy() if hasattr(out, "numpy") else out

    # ------------------------------------------------------------------ girdi
    def preprocess(self, img):
        """Cikis: (H,W,3) float32 [-1,1]. Pillow ile hafif ve hizli cozumleme."""
        if isinstance(img, (bytes, bytearray)):
            im = Image.open(io.BytesIO(img)).convert("RGB")
        elif isinstance(img, (str, os.PathLike)):
            im = Image.open(str(img)).convert("RGB")
        elif isinstance(img, np.ndarray):
            if img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] >= 3 \
                    and img.shape[:2] == self.img_size:
                return img[..., :3].astype(np.float32) / 127.5 - 1.0
            im = Image.fromarray(img).convert("RGB")
        else:
            raise TypeError(f"Desteklenmeyen gorsel tipi: {type(img)}")

        if im.size != (self.img_size[1], self.img_size[0]):
            im = im.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)

        arr = np.asarray(im, dtype=np.float32)
        return arr / 127.5 - 1.0

    def preprocess_many(self, imgs):
        """Cift islemi onler: halihazirda on islenmis dizileri tekrar isleme sokmaz."""
        out = []
        for i in imgs:
            if isinstance(i, np.ndarray) and i.shape == (self.img_size[0], self.img_size[1], 3) and i.dtype == np.float32:
                out.append(i)
            else:
                out.append(self.preprocess(i))
        return np.stack(out, axis=0)

    # ----------------------------------------------------------------- skorlar
    def scores_batch(self, imgs, return_embeddings=False):
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
        emb = self._forward(stacked)
        s = emb @ self.centers.T

        total = np.zeros((b, self.n), dtype=np.float32)
        for k, perm in enumerate(perms):
            part = s[k * b : (k + 1) * b]
            total += part if perm is None else part[:, perm]
        total /= float(len(views))
        return (total, emb[:b]) if return_embeddings else total

    # -------------------------------------------------------- tek gorsel / yon
    def predict_batch(self, imgs, apply_threshold=True):
        s = self.scores_batch(self.preprocess_many(imgs))
        out = []
        for row in s:
            cls = int(np.argmax(row))
            conf = float(row[cls])
            out.append((None, conf) if apply_threshold and conf < self.conf_threshold
                       else (cls, conf))
        return out

    def predict(self, img, apply_threshold=True):
        return self.predict_batch([img], apply_threshold)[0]

    def embeddings(self, imgs):
        _, emb = self.scores_batch(self.preprocess_many(imgs), return_embeddings=True)
        return emb

    # ------------------------------------------------------------ captcha cozme
    def solve_captcha(self, clue, candidates, min_margin=0.10):
        if not candidates:
            raise ValueError("candidates bos")

        imgs = [clue] + list(candidates)
        s, emb = self.scores_batch(self.preprocess_many(imgs), return_embeddings=True)
        p = softmax(self.scale * s, axis=1)

        agree = p[1:] @ p[0]
        dist = np.linalg.norm(emb[1:] - emb[0], axis=1)
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
# CLI & Benchmark
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Captcha yon cikarimi (ONNX & Keras)")
    p.add_argument("--meta", default="direction_embedding_v8_meta.json")
    p.add_argument("--image", help="tek goruntunun yonu")
    p.add_argument("--solve", nargs="+", metavar="YOL", help="clue.png opt1.png opt2.png ... (ilki clue)")
    p.add_argument("--tta-views", type=int, default=2, choices=(1, 2, 4))
    p.add_argument("--threads", type=int, default=2, help="CPU thread sayisi (default: 2)")
    p.add_argument("--min-margin", type=float, default=0.10)
    args = p.parse_args()

    clf = DirectionClassifier(args.meta, tta_views=args.tta_views, num_threads=args.threads)
    backend = "ONNX Runtime" if clf.is_onnx else "TensorFlow/Keras"
    print(f"Backend: {backend} ({os.path.basename(clf.model_file)}) | TTA views: {clf.tta_views} | Threads: {args.threads}")

    if args.image:
        t0 = time.perf_counter()
        cls, conf = clf.predict(args.image)
        dt = (time.perf_counter() - t0) * 1000
        label = "EMIN DEGIL" if cls is None else clf.class_names[cls]
        print(f"\n{args.image} -> {label} (conf={conf:.3f}) | {dt:.1f} ms")
    elif args.solve:
        if len(args.solve) < 2:
            p.error("--solve icin en az clue + 1 secenek gerekiyor")
        t0 = time.perf_counter()
        res = clf.solve_captcha(args.solve[0], args.solve[1:], min_margin=args.min_margin)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\nclue: {res['clue']['value']} -> {res['clue']['direction']} (conf={res['clue']['confidence']:.3f})")
        print("\nsecenekler (yon uyumuna gore sirali):")
        for i, c in enumerate(res["candidates"]):
            print(f"  {'-> ' if i == 0 else '   '}{c['score']:.3f}  {c['direction']:<14} {c['value']}")
        print(f"\nkazanan: {res['winner']['value']} (margin={res['margin']:.3f}) | {dt:.1f} ms")
    else:
        print("Lutfen --image veya --solve argumani verin.")


if __name__ == "__main__":
    main()
