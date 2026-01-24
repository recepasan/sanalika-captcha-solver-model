# Captcha Yön Sınıflandırma Modeli

Bu proje, captcha görsellerini 8 farklı yön sınıfına ayıran bir Siamese Neural Network modelidir. TensorFlow/Keras kullanılarak geliştirilmiştir.

## Proje Açıklaması

Model, captcha görsellerinin hareket yönünü tespit etmek için Contrastive Learning yaklaşımı kullanır. MobileNetV2 tabanlı bir embedding network ile görsel özelliklerini çıkarır ve Siamese Network mimarisi ile yön benzerliklerini öğrenir.

## Yön Sınıfları

Model 8 farklı yönü tespit edebilir:
- `0_up` - Yukarı
- `1_up_right` - Yukarı-Sağ
- `2_right` - Sağ
- `3_down_right` - Aşağı-Sağ
- `4_down` - Aşağı
- `5_down_left` - Aşağı-Sol
- `6_left` - Sol
- `7_up_left` - Yukarı-Sol

## Dataset

Dataset, 8 yön sınıfı için toplam **3,355 görsel** içermektedir (~17 MB).

### Dataset İndirme ve Hazırlama

1. `dataset.zip` dosyasını indirin
2. Proje kök dizinine çıkarın:
```bash
unzip dataset.zip
```

Dataset yapısı:
```
dataset/
├── 0_up/
├── 1_up_right/
├── 2_right/
├── 3_down_right/
├── 4_down/
├── 5_down_left/
├── 6_left/
└── 7_up_left/
```

## Kurulum

### Gereksinimler

```bash
pip install tensorflow numpy pillow matplotlib
```

### GPU Desteği (Opsiyonel)

NVIDIA GPU kullanıyorsanız:
```bash
pip install tensorflow[and-cuda]
```

## Kullanım

### Model Eğitimi

```bash
python model-egitim.py
```

### Örnek Kullanım (Standalone)

```bash
python example.py
```

```python
from example import load_model, get_embedding, calculate_distance, find_most_similar

# Modeli yükle
model = load_model("direction_embedding_best_windows.keras")

# İki görsel karşılaştır
emb1 = get_embedding(model, "gorsel1.png")
emb2 = get_embedding(model, "gorsel2.png")
mesafe = calculate_distance(emb1, emb2)

# Captcha çöz (clue'ya en benzer görseli bul)
result = find_most_similar(model, "clue.png", ["opt1.png", "opt2.png", "opt3.png"])
print(result["winner"])
```

### API ile Kullanım

```bash
python example-with-api.py
```

API `http://localhost:5000` üzerinde çalışır.

**Endpoint:** `POST /download-photo`

```bash
curl -X POST http://localhost:5000/download-photo \
  -H "Content-Type: application/json" \
  -d '{"clue": "abc123", "keys": ["key1", "key2", "key3"]}'
```

**Response:**
```json
{
  "success": true,
  "winner": {"key": "key2", "distance": 0.1234},
  "results": [...],
  "timing": {"total": 1.5, "download": 0.8, ...}
}
```

### Eğitim Parametreleri

Script içinde aşağıdaki parametreleri düzenleyebilirsiniz:

```python
IMG_SIZE = (128, 128)    # Görsel boyutu
BATCH = 64               # Batch size
EPOCHS = 100             # Epoch sayısı
MARGIN = 1.0             # Contrastive loss margin
```

### Çıktılar

Eğitim sonunda aşağıdaki dosyalar oluşturulur:
- `direction_embedding_best_windows.keras` - Eğitilmiş embedding modeli
- `direction_siamese_direction_embedding_best_windows.keras` - En iyi checkpoint
- `training_direction.png` - Eğitim grafikleri
- `logs_direction/` - TensorBoard logları

### TensorBoard ile Takip

```bash
tensorboard --logdir=logs_direction
```

## Model Mimarisi

### Embedding Network
- **Backbone:** MobileNetV2 (ImageNet ağırlıkları ile)
- **Embedding Boyutu:** 1024
- **Normalizasyon:** L2 normalization
- **Özellikler:**
  - Spatial Attention mekanizması
  - Batch Normalization
  - Dropout regularization (0.3-0.4)
  - L2 weight regularization

### Siamese Architecture
- **Loss Function:** Balanced Contrastive Loss
- **Distance Metric:** Euclidean Distance
- **Optimizer:** Adam (lr=2e-4)

## Eğitim Stratejisi

### Balanced Pair Generation
Model dengeli bir şekilde eğitilir:
1. **Pozitif Pairler:** Aynı yönden captcha görsel çiftleri
2. **Hard Negative Pairler:**
   - Zıt yönler (ağırlık: 2.0)
   - Ayna simetrisi (ağırlık: 2.0)
   - Komşu yönler (ağırlık: 1.8)
3. **Easy Negative Pairler:** Random farklı yönler (ağırlık: 1.0)

### Data Augmentation
Yön bilgisi korunarak:
- Brightness/Contrast/Saturation ayarları
- Random noise ekleme
- Hafif crop işlemleri
- Kontrollü flip (sadece negatif pairler için)

## Performans

Beklenen sonuçlar:
- **Validation Accuracy:** ~99%+
- **Loss:** ~0.05-0.15 arası
- **Training Time:** GPU ile yaklaşık 1-2 saat (100 epoch)

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## Katkıda Bulunma

Önerileriniz için issue açabilirsiniz.

## İletişim

Sorularınız için GitHub Issues kullanabilirsiniz.
