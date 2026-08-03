"""
Captcha yon siniflandirma modeli - egitim

Gorev: captcha gorsellerini 8 yon sinifina ayirmak.

Yaklasim
--------
Kapali kumeli 8 sinif oldugu icin pair uretimi yerine SINIFLANDIRMA +
acisal margin (CosFace) kullanilir. Cikti yine L2-normalize embedding
oldugu icin captcha cozme (clue'ya en benzer secenek) mantigi bozulmaz;
ayrica sinif merkezleri kaydedildigi icin tek gorselden dogrudan yon
tahmini de yapilabilir.

Onceki pair tabanli surumden farklar
------------------------------------
1) Split GORSEL seviyesinde. Onceki surumde split `pairs` listesi uzerinde
   yapiliyordu, bu yuzden ayni gorsel hem egitimde hem dogrulamada yer
   aliyor ve val skoru gercegi yansitmiyordu.
2) Yon simetrileri etiket yeniden haritalanarak augmentation olarak
   kullaniliyor:
      yatay cevir  : i -> (8 - i) % 8      (right <-> left)
      dikey cevir  : i -> (4 - i) % 8      (up    <-> down)
      90k derece   : i -> (i - 2k) % 8     (opsiyonel, --rot90)
   Onceki surumde negatif pairlerin bir gorseli cevriliyor ama etiket
   guncellenmiyordu; (2_right, 6_left) cifti cevrilince (right, right)
   olup "farkli yon" etiketiyle kaliyordu.
3) Fotometrik augmentation [0,1] araliginda yapiliyor. random_hue ve
   random_saturation RGB->HSV donusumu yapar ve [0,1] girdi bekler;
   onceki surumde [-1,1] araliginda cagriliyordu.
4) Backbone BatchNorm katmanlari daima inference modunda (training=False).
   Onceki surumde training=True sabitlenmisti, bu da dogrulama ve
   cikarimda farkli davranisa yol aciyordu.
5) Loss'a verilen sample_weight artik yok: Keras loss fonksiyonuna
   sample_weight gecirmez, eski koddaki o blok hic calismiyordu.
6) Rapor gercek metrikler uretiyor: 8-yon accuracy, dairesel hata dagilimi,
   hatali/sinirda gorsellerin listesi ve guven esigi kalibrasyonu.

Kullanim
--------
    python model-egitim.py
    python model-egitim.py --dataset dataset --rot90 --embed-dim 128
    python model-egitim.py --out direction_embedding_v2.keras

dataset/ altinda beklenen klasorler:
    0_up 1_up_right 2_right 3_down_right 4_down 5_down_left 6_left 7_up_left
"""

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, models
from tensorflow.keras.applications import MobileNetV2

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

# Saat yonunde, yukaridan baslayarak 45 derece adimlarla.
DIRECTION_ORDER = [
    "0_up",
    "1_up_right",
    "2_right",
    "3_down_right",
    "4_down",
    "5_down_left",
    "6_left",
    "7_up_left",
]


# ----------------------------------------------------------------------------
# Argumanlar / ortam
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Captcha yon siniflandirma egitimi")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs-head", type=int, default=6, help="backbone donuk asama")
    p.add_argument("--epochs-finetune", type=int, default=40)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--unfreeze-layers", type=int, default=60)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--scale", type=float, default=24.0, help="CosFace olcek (s)")
    p.add_argument("--margin", type=float, default=0.25, help="CosFace margin (m)")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-geom-aug", action="store_true", help="tum flip/rot augmentation kapat")
    p.add_argument("--no-hflip", action="store_true", help="sadece yatay cevirmeyi kapat")
    p.add_argument("--no-vflip", action="store_true", help="sadece dikey cevirmeyi kapat")
    p.add_argument("--rot90", action="store_true", help="90 derece donusleri de ekle")
    p.add_argument("--cache-file", default="", help="RAM yetmezse disk cache yolu")
    p.add_argument("--out", default="direction_embedding.keras")
    return p.parse_args()


def setup_runtime():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU bulunamadi, CPU ile devam.")
        return
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("GPU: %s" % gpus[0].name)
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 (loss/embedding float32)")
    except Exception as exc:  # pragma: no cover
        print("GPU ayari basarisiz: %s" % exc)


# ----------------------------------------------------------------------------
# Veri indeksi ve gorsel seviyesinde split
# ----------------------------------------------------------------------------
def load_index(dataset_dir):
    root = Path(dataset_dir)
    if not root.is_dir():
        raise SystemExit("dataset klasoru bulunamadi: %s" % root.resolve())

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    class_names, paths, labels = [], [], []
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        files = []
        for ext in exts:
            files.extend(sorted(class_dir.glob(ext)))
        if not files:
            print("  uyari: %s bos, atlaniyor" % class_dir.name)
            continue
        idx = len(class_names)
        class_names.append(class_dir.name)
        paths.extend(str(f) for f in files)
        labels.extend([idx] * len(files))

    if len(class_names) < 2:
        raise SystemExit("En az 2 sinif gerekli.")

    labels = np.asarray(labels, dtype=np.int32)
    print("Siniflar (%d):" % len(class_names))
    for i, name in enumerate(class_names):
        print("  [%d] %-14s %5d gorsel" % (i, name, int((labels == i).sum())))
    return np.asarray(paths), labels, class_names


def stratified_split(labels, val_frac):
    """Gorsel seviyesinde, sinif dengeli split. Ayni gorsel iki tarafta olamaz."""
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx), np.asarray(val_idx)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def decode(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)
    img.set_shape((img_size[0], img_size[1], 3))
    return img


def to_model_input(img_u8):
    return tf.cast(img_u8, tf.float32) / 127.5 - 1.0


def augment(img_u8, label, img_size, n_classes, hflip, vflip, rot):
    """Yon etiketini DOGRU sekilde yeniden haritalayan augmentation."""
    img = tf.cast(img_u8, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)

    if hflip:
        # Yatay cevirme: saat yonu acisi isaret degistirir -> i -> (n - i) % n
        do_h = tf.random.uniform([]) < 0.5
        img = tf.cond(do_h, lambda: tf.image.flip_left_right(img), lambda: img)
        label = tf.where(do_h, tf.math.floormod(n_classes - label, n_classes), label)

    if vflip:
        # Dikey cevirme: aci -> 180 - aci  ->  i -> (n/2 - i) % n
        do_v = tf.random.uniform([]) < 0.5
        img = tf.cond(do_v, lambda: tf.image.flip_up_down(img), lambda: img)
        half = n_classes // 2
        label = tf.where(do_v, tf.math.floormod(half - label, n_classes), label)

    if rot:
        # rot90(k) goruntuyu saat yonunun tersine cevirir -> i -> (i - 2k) % n
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)
        label = tf.math.floormod(label - 2 * k, n_classes)

    # Oteleme (yon-guvenli): reflect pad + random crop
    pad = tf.maximum(img_size[0] // 16, 2)
    img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, [img_size[0], img_size[1], 3])

    # Fotometrik: [0,1] araliginda olmali (hue/saturation RGB->HSV varsayar)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.image.random_hue(img, 0.03)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = img + tf.random.normal(tf.shape(img), stddev=0.01)
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img * 2.0 - 1.0, label


def build_dataset(
    paths,
    labels,
    img_size,
    batch,
    n_classes,
    class_weights=None,
    training=False,
    hflip=True,
    vflip=True,
    rot=False,
    cache_file="",
):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype(np.int32)))
    ds = ds.map(lambda p, y: (decode(p, img_size), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache(cache_file) if cache_file else ds.cache()

    if training:
        ds = ds.shuffle(min(len(paths), 8192), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: augment(x, y, img_size, n_classes, hflip, vflip, rot),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        ds = ds.map(lambda x, y: (to_model_input(x), y), num_parallel_calls=AUTOTUNE)

    if training and class_weights is not None:
        w = tf.constant(class_weights, dtype=tf.float32)
        ds = ds.map(
            lambda x, y: (x, tf.one_hot(y, n_classes), tf.gather(w, y)),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        # Validation'a sample_weight verilmez: val_loss karsilastirilabilir kalsin.
        ds = ds.map(
            lambda x, y: (x, tf.one_hot(y, n_classes)), num_parallel_calls=AUTOTUNE
        )

    return ds.batch(batch).prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def l2norm_layer(name="l2_norm"):
    """Ozel katman kullanmamak icin: mumkunse yerlesik UnitNormalization.

    Boylece kaydedilen .keras dosyasi custom_objects olmadan yuklenebilir.
    """
    if hasattr(layers, "UnitNormalization"):
        return layers.UnitNormalization(axis=-1, name=name, dtype="float32")
    return layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1), name=name, dtype="float32"
    )


class CosFaceHead(layers.Layer):
    """Cikis = cos(theta): L2-normalize embedding ile normalize sinif merkezleri.

    Margin loss tarafinda uygulanir, boylece bu katman etiket gormez ve
    accuracy metrigi dogrudan cos uzerinden hesaplanabilir.
    Ogrenilen agirliklar sinif merkezleri (centroid) olarak disa aktarilir.
    """

    def __init__(self, n_classes, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super(CosFaceHead, self).__init__(**kwargs)
        self.n_classes = int(n_classes)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="class_centers",
            shape=(int(input_shape[-1]), self.n_classes),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32",
        )
        super(CosFaceHead, self).build(input_shape)

    def call(self, inputs):
        x = tf.math.l2_normalize(tf.cast(inputs, tf.float32), axis=-1)
        w = tf.math.l2_normalize(self.kernel, axis=0)
        return tf.clip_by_value(tf.matmul(x, w), -1.0 + 1e-7, 1.0 - 1e-7)

    def centers(self):
        w = tf.math.l2_normalize(self.kernel, axis=0).numpy()
        return w.T  # (n_classes, embed_dim)

    def get_config(self):
        cfg = super(CosFaceHead, self).get_config()
        cfg["n_classes"] = self.n_classes
        return cfg


def build_models(img_size, embed_dim, n_classes, dropout):
    base = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inp = Input(shape=(img_size[0], img_size[1], 3), name="image")
    # training=False -> backbone BatchNorm daima inference modunda.
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    # BNNeck: metric learning'de standart, embedding'i merkezler ve olcekler.
    x = layers.Dense(embed_dim, use_bias=False, name="embed_dense")(x)
    x = layers.BatchNormalization(name="embed_bn")(x)
    x = l2norm_layer("l2_norm")(x)

    embed = models.Model(inp, x, name="direction_embedding")
    head = CosFaceHead(n_classes, name="cosface")
    clf = models.Model(inp, head(x), name="direction_classifier")
    return base, embed, clf, head


def cosface_loss(margin_var, scale, label_smoothing):
    def loss_fn(y_true, cos):
        y_true = tf.cast(y_true, cos.dtype)
        logits = scale * (cos - margin_var * y_true)
        return tf.keras.losses.categorical_crossentropy(
            y_true, logits, from_logits=True, label_smoothing=label_smoothing
        )

    return loss_fn


class MarginWarmup(tf.keras.callbacks.Callback):
    """Margin'i 0'dan hedefe rampalar; basta cokmeyi engeller."""

    def __init__(self, margin_var, target, epochs):
        super(MarginWarmup, self).__init__()
        self.margin_var = margin_var
        self.target = float(target)
        self.epochs = max(int(epochs), 1)

    def on_epoch_begin(self, epoch, logs=None):
        value = self.target * min(1.0, (epoch + 1) / float(self.epochs))
        self.margin_var.assign(value)


# ----------------------------------------------------------------------------
# Rapor
# ----------------------------------------------------------------------------
def circular_distance(a, b, n):
    d = np.abs(a - b)
    return np.minimum(d, n - d)


def dump_inspection(val_paths, val_labels, pred, cos, class_names,
                    out_dir="inspect", csv_path="predictions.csv",
                    n_borderline=15):
    """Hatali ve sinirda kalan gorselleri listeler, CSV yazar, gozle kontrol
    icin tek klasore kopyalar."""
    n = len(val_labels)
    order = np.argsort(-cos, axis=1)
    rows_i = np.arange(n)
    margin = cos[rows_i, order[:, 0]] - cos[rows_i, order[:, 1]]
    wrong = pred != val_labels

    rows = []
    for i in range(n):
        rows.append({
            "durum": "HATA" if wrong[i] else "dogru",
            "margin": round(float(margin[i]), 4),
            "gercek": class_names[val_labels[i]],
            "tahmin": class_names[pred[i]],
            "rakip": class_names[order[i, 1]],
            "cos_tahmin": round(float(cos[i, pred[i]]), 4),
            "cos_gercek": round(float(cos[i, val_labels[i]]), 4),
            "dosya": str(val_paths[i]),
        })
    # Once hatalar, sonra margin'i en dusuk olanlar
    rows.sort(key=lambda r: (r["durum"] != "HATA", r["margin"]))
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nHATALI GORSELLER (%d):" % int(wrong.sum()))
    if not wrong.any():
        print("  yok")
    for i in np.where(wrong)[0]:
        print("  gercek=%-14s tahmin=%-14s cos_tahmin=%.3f cos_gercek=%.3f"
              % (class_names[val_labels[i]], class_names[pred[i]],
                 cos[i, pred[i]], cos[i, val_labels[i]]))
        print("      %s" % val_paths[i])

    ok_idx = np.where(~wrong)[0]
    border = ok_idx[np.argsort(margin[ok_idx])][:n_borderline]
    print("\nDOGRU AMA SINIRDA (en dusuk margin %d gorsel):" % len(border))
    for i in border:
        print("  %-14s margin=%.3f  en_yakin_rakip=%-14s" % (
            class_names[val_labels[i]], margin[i], class_names[order[i, 1]]))
        print("      %s" % val_paths[i])

    # Gozle kontrol klasoru: sadece kendi urettigimiz dosyalari temizle
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("HATA__*")) + list(out.glob("sinirda__*")):
        try:
            old.unlink()
        except OSError:
            pass

    copied = 0
    for tag, idxs in (("HATA", np.where(wrong)[0]), ("sinirda", border)):
        for i in idxs:
            src = Path(val_paths[i])
            rival = class_names[pred[i]] if tag == "HATA" else class_names[order[i, 1]]
            name = "%s__gercek-%s__rakip-%s__m%.3f__%s" % (
                tag, class_names[val_labels[i]], rival, margin[i], src.name)
            try:
                shutil.copy2(src, out / name)
                copied += 1
            except OSError as exc:
                print("  kopyalanamadi: %s (%s)" % (src, exc))
    print("\n  %d gorsel kopyalandi -> %s/" % (copied, out))
    print("  tum tahminler (hatalar ustte, margin'e gore sirali) -> %s" % csv_path)


def _gate_table(name, score, correct):
    """Hedef kapsam seviyelerinde red esigi / accuracy tradeoff tablosu."""
    print("\n  %s tabanli red esigi:" % name)
    print("    esik    kapsam   kapsanan_acc  reddedilen_dogru")
    for cov_target in (1.00, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80):
        thr = float(np.quantile(score, 1.0 - cov_target))
        mask = score >= thr
        cov = float(mask.mean())
        acc = float(correct[mask].mean()) if mask.any() else 0.0
        red = int((~mask & correct).sum())
        print("    %5.3f  %6.2f%%   %7.2f%%      %4d"
              % (thr, 100.0 * cov, 100.0 * acc, red))


def confidence_gate(cos, pred, val_labels, safety=0.005):
    """Tahmin guveni uzerinden red esigi.

    Iki skor karsilastirilir:
      - cos      : secilen sinifin cos degeri (mutlak yakinlik)
      - margin   : en iyi ile ikinci sinif arasindaki fark (ayirt edicilik)
    Hatalarin tamamini reddeden en dusuk esik ve kapsam bedeli raporlanir.
    """
    idx = np.arange(len(pred))
    conf = cos[idx, pred]
    order = np.argsort(-cos, axis=1)
    margin = cos[idx, order[:, 0]] - cos[idx, order[:, 1]]
    correct = pred == val_labels

    print("\nGuven analizi:")
    ok = np.sort(conf[correct])
    print("  cos (dogru tahminler) : min=%.3f  p1=%.3f  p5=%.3f  ortalama=%.3f"
          % (ok[0], ok[max(len(ok) // 100, 0)], ok[max(len(ok) // 20, 0)], ok.mean()))
    if not (~correct).any():
        print("  hatali tahmin yok, esik gerekmiyor")
        return 0.0

    print("  cos (hatali tahminler): max=%.3f  (%d gorsel)"
          % (conf[~correct].max(), int((~correct).sum())))
    print("  margin (hatali)       : max=%.3f" % margin[~correct].max())

    _gate_table("cos", conf, correct)
    _gate_table("margin", margin, correct)

    thr = float(conf[~correct].max()) + safety
    mask = conf >= thr
    print("\n  TUM hatalari eleyen cos esigi: %.3f -> kapsam %.2f%%, "
          "kapsanan accuracy %.2f%%, bedel %d dogru tahmin"
          % (thr, 100.0 * mask.mean(),
             100.0 * (correct[mask].mean() if mask.any() else 0.0),
             int((~mask & correct).sum())))
    if ok[0] > conf[~correct].max():
        print("  AYRIM TAM: dogrularin en kotusu (%.3f) hatalarin en iyisinden "
              "(%.3f) yuksek" % (ok[0], conf[~correct].max()))
    return thr


def report(embed, val_paths, val_labels, centers, class_names, img_size, batch,
           prefix="direction_embedding"):
    n_classes = len(class_names)
    ds = (
        tf.data.Dataset.from_tensor_slices(val_paths)
        .map(lambda p: to_model_input(decode(p, img_size)), num_parallel_calls=AUTOTUNE)
        .batch(batch)
        .prefetch(AUTOTUNE)
    )
    emb = embed.predict(ds, verbose=0).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

    cos = emb @ centers.T
    pred = np.argmax(cos, axis=1)
    acc = float((pred == val_labels).mean())

    print("\n" + "=" * 62)
    print("VALIDATION RAPORU (gorsel seviyesinde ayrilmis, sizinti yok)")
    print("=" * 62)
    print("8-yon accuracy (en yakin sinif merkezi): %.2f%%  (%d gorsel)"
          % (acc * 100.0, len(val_labels)))

    print("\nSinif bazinda:")
    for c in range(n_classes):
        m = val_labels == c
        if m.sum():
            print("  %-14s %6.2f%%  (%d)"
                  % (class_names[c], 100.0 * (pred[m] == c).mean(), int(m.sum())))

    wrong = pred != val_labels
    print("\nHata tipleri (dairesel uzaklik):")
    if wrong.sum() == 0:
        print("  hata yok")
    else:
        cd = circular_distance(pred[wrong], val_labels[wrong], n_classes)
        etiket = {1: "komsu yon (45)", 2: "dik yon (90)", 3: "135 derece",
                  4: "zit yon (180)"}
        for d in sorted(set(cd.tolist())):
            print("  %-16s %d hata (%.1f%% tum hatalarin)"
                  % (etiket.get(int(d), "uzaklik %d" % d), int((cd == d).sum()),
                     100.0 * (cd == d).mean()))

    print("\nMatris (satir=gercek, kolon=tahmin):")
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(val_labels, pred):
        cm[t, p] += 1
    print("      " + " ".join("%5d" % c for c in range(n_classes)))
    for c in range(n_classes):
        print("  %2d  %s" % (c, " ".join("%5d" % v for v in cm[c])))

    dump_inspection(val_paths, val_labels, pred, cos, class_names,
                    out_dir="inspect_%s" % prefix,
                    csv_path="%s_predictions.csv" % prefix)
    conf_thr = confidence_gate(cos, pred, val_labels)

    # Pair kullanimi icin esik: L2-normalize embedding -> mesafe [0, 2]
    rng = np.random.default_rng(SEED)
    n = len(val_labels)
    n_pairs = min(200000, n * (n - 1) // 2)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    dist = np.linalg.norm(emb[i] - emb[j], axis=1)
    same = val_labels[i] == val_labels[j]

    print("\nMesafe dagilimi (ayni yon vs farkli yon):")
    print("  ayni  : mean=%.3f  std=%.3f  p95=%.3f"
          % (dist[same].mean(), dist[same].std(), np.percentile(dist[same], 95)))
    print("  farkli: mean=%.3f  std=%.3f  p05=%.3f"
          % (dist[~same].mean(), dist[~same].std(), np.percentile(dist[~same], 5)))

    best_thr, best_bacc = 0.0, 0.0
    for thr in np.arange(0.02, 2.0, 0.01):
        tpr = float((dist[same] < thr).mean())
        tnr = float((dist[~same] >= thr).mean())
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, float(thr)
    print("  onerilen esik: %.2f  (dengeli pair accuracy %.2f%%)"
          % (best_thr, best_bacc * 100.0))
    return acc, best_thr, conf_thr


def plot_history(histories, out_png="training.png"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib yok, grafik atlandi")
        return

    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, label in (("loss", "Train"), ("val_loss", "Validation")):
        if key in merged:
            axes[0].plot(merged[key], label=label, linewidth=2)
    axes[0].set_title("CosFace Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key, label in (("accuracy", "Train"), ("val_accuracy", "Validation")):
        if key in merged:
            axes[1].plot(merged[key], label=label, linewidth=2)
    axes[1].set_title("8-Yon Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print("Grafik: %s" % out_png)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_runtime()

    img_size = (args.img_size, args.img_size)
    paths, labels, class_names = load_index(args.dataset)
    n_classes = len(class_names)

    geom_ok = class_names == DIRECTION_ORDER
    if not geom_ok:
        print(
            "\nUYARI: sinif isimleri beklenen sirada degil, geometrik augmentation "
            "(flip/rot) KAPATILDI.\n  beklenen: %s" % DIRECTION_ORDER
        )
    geom = geom_ok and not args.no_geom_aug
    hflip = geom and not args.no_hflip
    vflip = geom and not args.no_vflip
    rot = args.rot90 and geom

    tr, va = stratified_split(labels, args.val_frac)
    print("\nSplit (gorsel seviyesinde): train=%d  val=%d" % (len(tr), len(va)))

    counts = np.bincount(labels[tr], minlength=n_classes).astype(np.float32)
    class_weights = np.where(counts > 0, counts.mean() / np.maximum(counts, 1.0), 0.0)
    print("Sinif agirliklari: %s"
          % np.array2string(class_weights, precision=2, floatmode="fixed"))
    print("Augmentation -> yatay cevirme: %s | dikey cevirme: %s | rot90: %s"
          % ("acik" if hflip else "kapali",
             "acik" if vflip else "kapali",
             "acik" if rot else "kapali"))

    train_ds = build_dataset(
        paths[tr], labels[tr], img_size, args.batch, n_classes,
        class_weights=class_weights, training=True,
        hflip=hflip, vflip=vflip, rot=rot,
        cache_file=(args.cache_file + ".train") if args.cache_file else "",
    )
    val_ds = build_dataset(
        paths[va], labels[va], img_size, args.batch, n_classes, training=False,
        cache_file=(args.cache_file + ".val") if args.cache_file else "",
    )

    base, embed, clf, head = build_models(
        img_size, args.embed_dim, n_classes, args.dropout
    )
    print("\nEmbedding boyutu: %d | backbone: MobileNetV2" % args.embed_dim)
    print("Egitilebilir parametre (asama 1): %d"
          % int(sum(np.prod(w.shape) for w in clf.trainable_weights)))

    margin_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="margin")
    prefix = Path(args.out).stem
    ckpt = "%s_best.weights.h5" % prefix
    centers_path = "%s_centers.npy" % prefix
    meta_path = "%s_meta.json" % prefix
    plot_path = "%s_training.png" % prefix
    print("Cikti onendeki: %s_*  (checkpoint, centers, meta, grafik)" % prefix)
    steps = int(math.ceil(len(tr) / float(args.batch)))
    histories = []

    def compile_model(lr_schedule):
        clf.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=cosface_loss(margin_var, args.scale, args.label_smoothing),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
            weighted_metrics=[],
        )

    # ---- Asama 1: backbone donuk, sadece head ----
    if args.epochs_head > 0:
        print("\n>>> Asama 1/2: backbone donuk, head egitiliyor")
        compile_model(args.lr_head)
        histories.append(
            clf.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs_head,
                callbacks=[MarginWarmup(margin_var, args.margin, args.epochs_head)],
                verbose=1,
            )
        )

    # ---- Asama 2: ust katmanlari ac, cosine LR ----
    print("\n>>> Asama 2/2: backbone son %d katman aciliyor" % args.unfreeze_layers)
    base.trainable = True
    for layer in base.layers[: max(len(base.layers) - args.unfreeze_layers, 0)]:
        layer.trainable = False
    # BatchNorm'lar zaten training=False ile cagriliyor; agirliklarini da donduruyoruz.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    margin_var.assign(args.margin)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        args.lr_finetune, decay_steps=max(steps * args.epochs_finetune, 1), alpha=0.01
    )
    compile_model(schedule)
    print("Egitilebilir parametre (asama 2): %d"
          % int(sum(np.prod(w.shape) for w in clf.trainable_weights)))

    histories.append(
        clf.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_finetune,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    ckpt, monitor="val_accuracy", mode="max",
                    save_best_only=True, save_weights_only=True, verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", mode="max", patience=12,
                    restore_best_weights=True, verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(log_dir="logs_direction"),
            ],
            verbose=1,
        )
    )

    if Path(ckpt).exists():
        clf.load_weights(ckpt)
        print("\nEn iyi agirliklar geri yuklendi: %s" % ckpt)

    centers = head.centers()
    acc, thr, conf_thr = report(
        embed, paths[va], labels[va], centers, class_names, img_size, args.batch,
        prefix=prefix,
    )

    embed.save(args.out)
    np.save(centers_path, centers)
    meta = {
        "embedding_model": args.out,
        "centers": centers_path,
        "class_names": class_names,
        "embed_dim": int(args.embed_dim),
        "img_size": list(img_size),
        "preprocess": "resize -> float32/127.5 - 1.0",
        "embedding_normalized": True,
        "cosface_scale": float(args.scale),
        "cosface_margin": float(args.margin),
        "recommended_pair_threshold": round(float(thr), 3),
        "recommended_confidence_threshold": round(float(conf_thr), 3),
        "val_direction_accuracy": round(float(acc), 4),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    plot_history(histories, plot_path)

    print("\n" + "=" * 62)
    print("Kaydedilenler")
    print("=" * 62)
    print("  %-38s embedding modeli" % args.out)
    print("  %-38s sinif merkezleri" % centers_path)
    print("  %-38s esik + metadata" % meta_path)
    print("  %-38s hatali/sinirda gorseller" % ("inspect_%s/" % prefix))
    print("  %-38s tum val tahminleri" % ("%s_predictions.csv" % prefix))
    print("\nSonraki adim - esigi TTA ile kalibre et:")
    print("  python direction_infer.py --meta %s \\" % meta_path)
    print("      --eval %s_predictions.csv --tta-views 4 --save-threshold" % prefix)
    print("\nCaptcha cozme:")
    print("  from direction_infer import DirectionClassifier")
    print("  clf = DirectionClassifier('%s', tta_views=4)" % meta_path)
    print("  sonuc = clf.solve_captcha('clue.png', ['opt1.png', 'opt2.png', ...])")
    print("\nTek gorselden yon:")
    print("  cls, conf = clf.predict('gorsel.png')   # conf < %.3f ise emin degil"
          % conf_thr)
    print("Pair karsilastirmasi icin esik: %.2f (mesafe < esik => ayni yon)" % thr)


if __name__ == "__main__":
    main()
