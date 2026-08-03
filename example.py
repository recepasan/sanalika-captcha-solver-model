"""
Captcha yon siniflandirma - ornek kullanim
==========================================
Egitilmis modeli kullanarak:
  1) tek gorselin yonunu bulur,
  2) iki gorselin ayni yonu gosterip gostermedigini soyler,
  3) captcha cozer (clue ile ayni yonu gosteren secenegi bulur).

Gerekli dosyalar (model-egitim.py uretir):
    direction_embedding.keras
    direction_embedding_centers.npy
    direction_embedding_meta.json

Girdi : 128x128 RGB gorsel        Cikti: 128 boyutlu L2-normalize embedding

DIKKAT - renk sirasi
--------------------
Model RGB ile egitildi (tf.image.decode_image RGB uretir). cv2.imread BGR
dondurur; cv2 kullaniyorsaniz mutlaka cevirin:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
Bu modul dosya yollarini dogrudan TensorFlow ile okudugu icin sorun yasamaz.
"""

import os

import numpy as np

from direction_infer import DirectionClassifier

# ============================================
# AYARLAR
# ============================================
META_PATH = "direction_embedding_meta.json"
TTA_VIEWS = 4       # 1, 2 veya 4. Yuksek = daha dogru, daha yavas.
DATASET_DIR = "dataset"


# ============================================
# MODEL YUKLEME
# ============================================
def load_model(meta_path=META_PATH, tta_views=TTA_VIEWS):
    """Modeli, sinif merkezlerini ve esikleri yukler."""
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Metadata bulunamadi: %s\nOnce model-egitim.py ile modeli egitin."
            % meta_path
        )
    print("Model yukleniyor: %s" % meta_path)
    clf = DirectionClassifier(meta_path, tta_views=tta_views)
    print("Model yuklendi. Siniflar: %s" % ", ".join(clf.class_names))
    print("Guven esigi: %.3f | TTA views: %d" % (clf.conf_threshold, clf.tta_views))
    return clf


# ============================================
# YON TAHMINI
# ============================================
def classify(clf, image_path):
    """Tek gorselin yonunu dondurur.

    Cikis: (yon_adi | None, guven). yon_adi None ise model emin degil,
    yeni bir goruntu almak yanlis cevap vermekten iyidir.
    """
    cls, conf = clf.predict(image_path)
    return (None if cls is None else clf.class_names[cls]), conf


# ============================================
# EMBEDDING / MESAFE (geriye donuk uyumluluk)
# ============================================
def get_embedding(clf, image_path):
    """Tek gorsel icin embedding vektoru."""
    return clf.embeddings([image_path])[0]


def get_embeddings_batch(clf, image_paths):
    """Birden fazla gorsel icin embedding matrisi."""
    if not image_paths:
        return np.array([])
    return clf.embeddings(image_paths)


def calculate_distance(embedding1, embedding2):
    """Iki embedding arasindaki Oklid mesafesi (L2-normalize => [0, 2])."""
    return float(np.linalg.norm(np.asarray(embedding1) - np.asarray(embedding2)))


def is_same_direction(clf, path1, path2):
    """Iki gorsel ayni yonu mu gosteriyor? (yon siniflandirmasiyla karar)"""
    (a, ca), (b, cb) = clf.predict_batch([path1, path2])
    if a is None or b is None:      # en az biri belirsiz
        return None, min(ca, cb)
    return a == b, min(ca, cb)


# ============================================
# CAPTCHA COZME
# ============================================
def find_most_similar(clf, clue_path, candidate_paths, min_margin=0.10):
    """clue ile ayni yonu gosteren secenegi bulur.

    Skorlama yon uyumu olasiligi uzerinden yapilir; ham embedding mesafesi
    de raporlanir. Donen sozluk eski surumle uyumludur ("winner",
    "all_results", her kayitta "path" ve "distance").
    """
    res = clf.solve_captcha(clue_path, candidate_paths, min_margin=min_margin)
    for c in res["candidates"]:
        c["path"] = c["value"]
    res["winner"]["path"] = res["winner"]["value"]
    res["all_results"] = res["candidates"]
    return res


# ============================================
# ORNEK KULLANIM
# ============================================
def _images_in(class_dir):
    return sorted(
        f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def main():
    try:
        clf = load_model()
    except FileNotFoundError as exc:
        print("HATA: %s" % exc)
        return 1

    if not os.path.isdir(DATASET_DIR):
        print("\n%s klasoru yok. Kendi gorsellerinizle:" % DATASET_DIR)
        print("  yon, guven = classify(clf, 'gorsel.png')")
        print("  sonuc = find_most_similar(clf, 'clue.png', ['a.png', 'b.png'])")
        return 0

    classes = sorted(d for d in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, d)))
    if len(classes) < 3:
        print("Ornekler icin en az 3 sinif gerekiyor.")
        return 0

    # ---- 1) Tek gorselin yonu ----
    print("\n" + "=" * 55)
    print("ORNEK 1: Tek gorselin yonu")
    print("=" * 55)
    for cls in classes[:3]:
        d = os.path.join(DATASET_DIR, cls)
        imgs = _images_in(d)
        if not imgs:
            continue
        path = os.path.join(d, imgs[0])
        yon, conf = classify(clf, path)
        print("  %-40s -> %-14s (conf=%.3f)"
              % (os.path.basename(path), yon or "EMIN DEGIL", conf))
        print("     gercek etiket: %s" % cls)

    # ---- 2) Iki gorsel karsilastirma ----
    print("\n" + "=" * 55)
    print("ORNEK 2: Iki gorsel ayni yonu mu gosteriyor?")
    print("=" * 55)
    d0 = os.path.join(DATASET_DIR, classes[0])
    d1 = os.path.join(DATASET_DIR, classes[1])
    i0, i1 = _images_in(d0), _images_in(d1)
    if len(i0) >= 2 and i1:
        ayni = os.path.join(d0, i0[0]), os.path.join(d0, i0[1])
        farkli = os.path.join(d0, i0[0]), os.path.join(d1, i1[0])
        for etiket, (p, q) in (("ayni sinif", ayni), ("farkli sinif", farkli)):
            karar, conf = is_same_direction(clf, p, q)
            emb = get_embeddings_batch(clf, [p, q])
            print("  %-13s -> ayni yon: %-9s mesafe=%.3f  (min conf=%.3f)"
                  % (etiket,
                     {True: "EVET", False: "HAYIR", None: "BELIRSIZ"}[karar],
                     calculate_distance(emb[0], emb[1]), conf))

    # ---- 3) Captcha cozme ----
    print("\n" + "=" * 55)
    print("ORNEK 3: Captcha cozme (clue + secenekler)")
    print("=" * 55)
    clue_cls = classes[0]
    clue_dir = os.path.join(DATASET_DIR, clue_cls)
    clue_imgs = _images_in(clue_dir)
    if len(clue_imgs) < 2:
        print("Yeterli gorsel yok.")
        return 0

    clue_path = os.path.join(clue_dir, clue_imgs[0])
    dogru_cevap = os.path.join(clue_dir, clue_imgs[1])
    secenekler = [dogru_cevap]
    for cls in classes[1:4]:
        d = os.path.join(DATASET_DIR, cls)
        imgs = _images_in(d)
        if imgs:
            secenekler.append(os.path.join(d, imgs[0]))

    print("  clue        : %s  (gercek yon: %s)" % (os.path.basename(clue_path), clue_cls))
    print("  dogru cevap : %s" % os.path.basename(dogru_cevap))
    print("  secenek     : %d adet" % len(secenekler))

    sonuc = find_most_similar(clf, clue_path, secenekler)
    print("\n  clue tahmini: %s (conf=%.3f%s)"
          % (sonuc["clue"]["direction"], sonuc["clue"]["confidence"],
             "" if sonuc["clue"]["certain"] else ", EMIN DEGIL"))
    print("\n  secenekler (yon uyumuna gore sirali):")
    for i, c in enumerate(sonuc["all_results"]):
        print("    %sskor=%.3f  yon=%-14s mesafe=%.3f  %s"
              % ("-> " if i == 0 else "   ", c["score"], c["direction"],
                 c["distance"], os.path.basename(c["path"])))

    kazanan = sonuc["winner"]["path"]
    print("\n  kazanan: %s" % os.path.basename(kazanan))
    print("  margin : %.3f%s" % (sonuc["margin"],
                                 "  [YENIDEN DENE]" if sonuc["needs_retry"] else ""))
    print("  sonuc  : %s" % ("DOGRU" if kazanan == dogru_cevap else "YANLIS"))

    print("\n" + "=" * 55)
    print("TAMAMLANDI")
    print("=" * 55)
    print("\nUctan uca dogrulugu olcmek icin:")
    print("  python direction_infer.py --meta %s \\" % META_PATH)
    print("      --eval direction_embedding_predictions.csv --simulate 300")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
