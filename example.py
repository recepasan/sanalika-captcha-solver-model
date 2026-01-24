"""
Yön Sınıflandırma Modeli - Örnek Kullanım
=========================================
Eğitilmiş Siamese Network modelini kullanarak
iki görsel arasındaki benzerliği hesaplar.

Model: direction_embedding_best_windows.keras
Girdi: 128x128 RGB görsel
Çıktı: 1024 boyutlu embedding vektörü
"""

import tensorflow as tf
import numpy as np
import cv2
import os

# ============================================
# AYARLAR
# ============================================
MODEL_PATH = "direction_embedding_best_windows.keras"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5  # Bu değerin altındaki mesafeler "aynı yön" olarak kabul edilir


# ============================================
# MODEL YÜKLEME
# ============================================
def load_model(model_path):
    """Embedding modelini yükler"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    print(f"Model yükleniyor: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "lambda": lambda x: tf.math.l2_normalize(x, axis=1),
            "Lambda": lambda x: tf.math.l2_normalize(x, axis=1)
        },
        safe_mode=False
    )
    print("Model yüklendi!")
    return model


# ============================================
# GÖRSEL İŞLEME
# ============================================
def preprocess_image(image_path):
    """
    Görseli model için hazırlar.
    - 128x128 boyutuna resize
    - -1 ile 1 arasında normalize
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görsel okunamadı: {image_path}")

    # BGR -> RGB (opsiyonel, model BGR ile de çalışabilir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # Normalize: [0, 255] -> [-1, 1]
    img = img.astype("float32") / 127.5 - 1.0

    return img


def preprocess_batch(image_paths):
    """Birden fazla görseli toplu işler"""
    batch = []
    for path in image_paths:
        try:
            img = preprocess_image(path)
            batch.append(img)
        except Exception as e:
            print(f"Hata ({path}): {e}")
    return np.array(batch)


# ============================================
# EMBEDDİNG VE MESAFE HESAPLAMA
# ============================================
def get_embedding(model, image_path):
    """Tek bir görsel için embedding vektörü hesaplar"""
    img = preprocess_image(image_path)
    img_batch = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)
    embedding = model.predict(img_batch, verbose=0)
    return embedding[0]  # (1024,)


def get_embeddings_batch(model, image_paths):
    """Birden fazla görsel için embedding vektörleri hesaplar"""
    batch = preprocess_batch(image_paths)
    if len(batch) == 0:
        return np.array([])
    embeddings = model.predict(batch, verbose=0, batch_size=32)
    return embeddings


def calculate_distance(embedding1, embedding2):
    """İki embedding arasındaki Öklid mesafesini hesaplar"""
    return float(np.linalg.norm(embedding1 - embedding2))


def find_most_similar(model, clue_path, candidate_paths):
    """
    Clue görseline en benzer candidate'ı bulur.

    Args:
        model: Yüklenmiş embedding modeli
        clue_path: Referans görsel yolu
        candidate_paths: Karşılaştırılacak görsellerin yolları

    Returns:
        dict: En benzer görselin bilgileri
    """
    # Tüm görselleri tek batch'te işle (daha hızlı)
    all_paths = [clue_path] + candidate_paths
    embeddings = get_embeddings_batch(model, all_paths)

    if len(embeddings) < 2:
        raise ValueError("Yeterli görsel işlenemedi")

    clue_embedding = embeddings[0]

    results = []
    for i, path in enumerate(candidate_paths):
        dist = calculate_distance(clue_embedding, embeddings[i + 1])
        results.append({
            "path": path,
            "distance": dist,
            "is_same_direction": dist < THRESHOLD
        })

    # En düşük mesafe = en benzer
    results.sort(key=lambda x: x["distance"])

    return {
        "winner": results[0],
        "all_results": results
    }


# ============================================
# ÖRNEK KULLANIM
# ============================================
if __name__ == "__main__":
    # 1. Modeli yükle
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"HATA: {e}")
        print("Önce model-egitim.py ile modeli eğitin.")
        exit(1)

    # 2. Örnek: İki görsel karşılaştır
    print("\n" + "="*50)
    print("ÖRNEK 1: İki görsel karşılaştırma")
    print("="*50)

    # Dataset klasöründen örnek görseller (varsa)
    dataset_dir = "dataset"
    if os.path.exists(dataset_dir):
        # İlk iki sınıftan birer görsel al
        classes = sorted(os.listdir(dataset_dir))
        if len(classes) >= 2:
            class1_dir = os.path.join(dataset_dir, classes[0])
            class2_dir = os.path.join(dataset_dir, classes[1])

            img1_list = [f for f in os.listdir(class1_dir) if f.endswith(('.png', '.jpg'))]
            img2_list = [f for f in os.listdir(class2_dir) if f.endswith(('.png', '.jpg'))]

            if img1_list and img2_list:
                img1_path = os.path.join(class1_dir, img1_list[0])
                img2_path = os.path.join(class2_dir, img2_list[0])

                emb1 = get_embedding(model, img1_path)
                emb2 = get_embedding(model, img2_path)

                dist = calculate_distance(emb1, emb2)

                print(f"Görsel 1: {img1_path}")
                print(f"Görsel 2: {img2_path}")
                print(f"Mesafe: {dist:.4f}")
                print(f"Aynı yön mü?: {'Evet' if dist < THRESHOLD else 'Hayır'}")
    else:
        print("Dataset klasörü bulunamadı.")
        print("Kendi görsellerinizle test etmek için:")
        print("  emb = get_embedding(model, 'gorsel.png')")

    # 3. Örnek: Clue + Candidates (Captcha çözme senaryosu)
    print("\n" + "="*50)
    print("ÖRNEK 2: Captcha çözme senaryosu")
    print("="*50)

    if os.path.exists(dataset_dir):
        classes = sorted(os.listdir(dataset_dir))
        if len(classes) >= 3:
            # Clue: İlk sınıftan bir görsel
            clue_class = classes[0]
            clue_dir = os.path.join(dataset_dir, clue_class)
            clue_images = [f for f in os.listdir(clue_dir) if f.endswith(('.png', '.jpg'))]

            if clue_images:
                clue_path = os.path.join(clue_dir, clue_images[0])

                # Candidates: Aynı sınıftan + farklı sınıflardan görseller
                candidates = []

                # Aynı sınıftan bir tane (doğru cevap olması gereken)
                if len(clue_images) > 1:
                    candidates.append(os.path.join(clue_dir, clue_images[1]))

                # Farklı sınıflardan
                for cls in classes[1:4]:  # 3 farklı sınıf
                    cls_dir = os.path.join(dataset_dir, cls)
                    cls_images = [f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg'))]
                    if cls_images:
                        candidates.append(os.path.join(cls_dir, cls_images[0]))

                if candidates:
                    print(f"Clue: {clue_path}")
                    print(f"Candidates: {len(candidates)} adet")

                    result = find_most_similar(model, clue_path, candidates)

                    print(f"\nSonuçlar (mesafeye göre sıralı):")
                    for i, r in enumerate(result["all_results"]):
                        marker = " <-- EN BENZER" if i == 0 else ""
                        print(f"  {r['distance']:.4f} - {os.path.basename(r['path'])}{marker}")

                    print(f"\nKazanan: {result['winner']['path']}")
                    print(f"Mesafe: {result['winner']['distance']:.4f}")
    else:
        print("Dataset klasörü bulunamadı.")
        print("\nKullanım:")
        print("  result = find_most_similar(model, 'clue.png', ['opt1.png', 'opt2.png', 'opt3.png'])")
        print("  print(result['winner'])")

    print("\n" + "="*50)
    print("TAMAMLANDI")
    print("="*50)
