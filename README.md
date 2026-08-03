# Captcha Yön Sınıflandırma Modeli

Captcha görsellerini 8 yön sınıfına ayıran bir model. Asıl görev captcha çözmek: bir ipucu (clue) görseli ile aynı yönü gösteren seçeneği bulmak.

Yaklaşım, kapalı kümeli 8 sınıf için **sınıflandırma + açısal margin (CosFace)**. Çıktı yine L2-normalize embedding olduğu için mesafe tabanlı karşılaştırma da çalışır, ama sınıf merkezleri kaydedildiği için tek görselden doğrudan yön tahmini de yapılabilir.

## Yön Sınıfları

| İndeks | Sınıf | Yön |
|---|---|---|
| 0 | `0_up` | Yukarı |
| 1 | `1_up_right` | Yukarı-Sağ |
| 2 | `2_right` | Sağ |
| 3 | `3_down_right` | Aşağı-Sağ |
| 4 | `4_down` | Aşağı |
| 5 | `5_down_left` | Aşağı-Sol |
| 6 | `6_left` | Sol |
| 7 | `7_up_left` | Yukarı-Sol |

Sınıf indeksleri saat yönünde 45° adımlarla ilerler. Bu sıralama tesadüfi değil; eğitimde yön simetrilerini augmentation olarak kullanmayı mümkün kılıyor (aşağıya bakın).

## Kurulum

```bash
pip install tensorflow numpy matplotlib flask requests
unzip dataset.zip
```

GPU için `pip install "tensorflow[and-cuda]"`. Dataset 8 sınıf, toplam **3.355 görsel** (~17 MB).

```
dataset/
├── 0_up/  ├── 1_up_right/  ├── 2_right/  ├── 3_down_right/
├── 4_down/  ├── 5_down_left/  ├── 6_left/  └── 7_up_left/
```

## Kullanım

### 1. Eğitim

```bash
python model-egitim.py
```

Öne çıkan seçenekler:

```bash
python model-egitim.py --dataset dataset --embed-dim 128 --epochs-finetune 40
python model-egitim.py --rot90              # 90° dönüşleri de augmentation'a ekle
python model-egitim.py --no-vflip           # dikey çevirmeyi kapat
python model-egitim.py --out model_v2.keras # tüm çıktı adları bundan türer
```

Üretilen dosyalar (`--out` adından türetilir, eski koşuların üzerine yazılmaz):

| Dosya | İçerik |
|---|---|
| `direction_embedding.keras` | Embedding modeli (L2-normalize çıkış) |
| `direction_embedding_centers.npy` | Sınıf merkezleri (8 × embed_dim) |
| `direction_embedding_meta.json` | Eşikler, sınıf adları, ön işleme bilgisi |
| `direction_embedding_predictions.csv` | Tüm doğrulama tahminleri, hatalar üstte |
| `inspect_direction_embedding/` | Hatalı ve sınırda kalan görsellerin kopyaları |
| `direction_embedding_training.png` | Loss / accuracy grafikleri |
| `logs_direction/` | TensorBoard logları |

### 2. Eşik kalibrasyonu (eğitimden sonra bir kez)

```bash
python direction_infer.py --meta direction_embedding_meta.json \
    --eval direction_embedding_predictions.csv --tta-views 4 --save-threshold
```

TTA skor dağılımını değiştirdiği için güven eşiği yeniden hesaplanır ve `meta.json` içine yazılır. Her TTA view sayısı kendi eşiğini gerektirir.

### 3. Uçtan uca doğruluk (asıl metrik)

```bash
python direction_infer.py --meta direction_embedding_meta.json \
    --eval direction_embedding_predictions.csv --simulate 300 --options 4
```

Doğrulama görsellerinden sentetik captcha görevleri kurar (1 doğru + 3 yanlış seçenek) ve "doğru seçeneği seçtik mi" oranını ölçer. Görsel başına accuracy değil, ürünün gerçek başarı ölçütü bu.

### 4. Captcha çözme (kod içinde)

```python
from direction_infer import DirectionClassifier

clf = DirectionClassifier("direction_embedding_meta.json", tta_views=4)

sonuc = clf.solve_captcha("clue.png", ["a.png", "b.png", "c.png", "d.png"])

print(sonuc["winner"]["value"])        # kazanan seçenek
print(sonuc["clue"]["direction"])      # clue'nun yönü, örn. '2_right'
print(sonuc["margin"])                 # 1. ile 2. arasındaki fark
if sonuc["needs_retry"]:
    ...                                # emin değil: yeni captcha istemek daha ucuz
```

Girdi olarak dosya yolu, `bytes` (indirilen görsel içeriği) veya `HxWx3 uint8` RGB dizi verebilirsiniz.

### 5. Tek görselin yönü

```python
cls, conf = clf.predict("gorsel.png")
if cls is None:
    ...                                # güven eşiğinin altında, emin değil
else:
    yon = clf.class_names[cls]
```

### 6. Örnekler ve API

```bash
python example.py                      # 3 demo: yön, karşılaştırma, captcha çözme
python example-with-api.py             # Flask servisi (varsayılan 127.0.0.1:5000)
```

```bash
curl -X POST http://127.0.0.1:5000/solve \
  -H 'Content-Type: application/json' \
  -d '{"clue": "abc123", "keys": ["k1", "k2", "k3"]}'
```

```json
{
  "success": true,
  "winner": {"key": "k2", "score": 0.9871, "direction": "2_right", "distance": 0.31},
  "clue": {"key": "abc123", "direction": "2_right", "confidence": 0.83, "certain": true},
  "margin": 0.9712,
  "needs_retry": false,
  "results": [...],
  "timing": {"total_ms": 214.3, "download_ms": 198.1, "inference_ms": 13.2}
}
```

Diğer endpointler: `POST /direction` (tek görselin yönü), `GET /health`.

> **Güvenlik.** API kimlik doğrulaması içermez ve varsayılan olarak yalnızca `127.0.0.1`'e bağlanır. Ağa açmak için `--host 0.0.0.0` gerekir; bunu yapacaksanız `CAPTCHA_API_KEY` ortam değişkenini ayarlayın, ayarlıysa her istekte `X-API-Key` başlığı zorunlu olur. Kimlik doğrulaması olmadan ağa açarsanız ağdaki herkes servisi kullanabilir.

## Model Mimarisi

```
Görsel (128×128×3)
  → MobileNetV2 (ImageNet, BatchNorm daima inference modunda)
  → GlobalAveragePooling
  → Dropout(0.3)
  → Dense(128, bias yok) → BatchNorm → L2 normalize     ← embedding
  → CosFace başlığı: cos(θ) = normalize(embedding) · normalize(sınıf merkezleri)
```

Kayıp: `CE(softmax(s · (cosθ - m · onehot)))`, s = 24, m = 0.25 (ilk epoch'larda 0'dan rampalanır). Sınıflandırma başlığının ağırlıkları doğrudan sınıf merkezleri olarak dışa aktarılır.

Eğitim iki aşamalı: önce backbone donuk sadece başlık (6 epoch, lr 1e-3), sonra son 60 katman açık cosine decay ile fine-tune (40 epoch, lr 1e-4).

## Eğitim Stratejisi

### Görsel seviyesinde bölme

Doğrulama seti sınıf bazında dengeli, **görsel** seviyesinde ayrılır. Bir görsel ya eğitimde ya doğrulamada bulunur, ikisinde birden asla.

> Önceki sürüm bölmeyi çift (pair) listesi üzerinde yapıyordu; aynı görsel her iki tarafta yer aldığı için raporlanan ~%99 doğrulama skoru modelin genelleme yeteneğini ölçmüyordu.

### Yön simetrileri ile augmentation

Çevirme işlemi yönü değiştirir, dolayısıyla etiketin de değişmesi gerekir:

| Dönüşüm | Etiket eşlemesi | Örnek |
|---|---|---|
| Yatay çevirme | `i → (8 - i) % 8` | `2_right` → `6_left` |
| Dikey çevirme | `i → (4 - i) % 8` | `0_up` → `4_down` |
| 90°·k dönüş (`--rot90`) | `i → (i - 2k) % 8` | `0_up` → `6_left` (k=1) |

İki çevirme kombinasyonu ile her görselden 4, `--rot90` ile 16 geçerli varyant elde edilir. Etiket doğru haritalandığı için bu bedava veridir.

> Önceki sürümde negatif çiftlerin bir görseli çevriliyor ama etiket güncellenmiyordu. `(2_right, 6_left)` çifti çevrildiğinde `(right, right)` oluyor ve "farklı yön" etiketiyle kalıyordu.

### Fotometrik augmentation

Parlaklık, kontrast, doygunluk, ton ve hafif gürültü `[0, 1]` aralığında uygulanır, normalizasyon en sonda yapılır. `random_hue` ve `random_saturation` RGB→HSV dönüşümü yaptığı için `[0, 1]` girdi bekler.

Öteleme, reflect padding + rastgele kırpma ile yapılır — yön bilgisini bozmaz.

## Değerlendirme

Eğitim sonunda rapor şunları verir:

- **8-yön accuracy**, sınıf bazında dağılımıyla
- **Dairesel hata dağılımı**: komşu yön (45°) / dik (90°) / 135° / zıt yön (180°). Hataların komşu yönlerde toplanması sağlıklıdır; zıt yön hataları temel bir sorunu işaret eder.
- **Karışıklık matrisi**
- **Hatalı görsellerin listesi** dosya yollarıyla, ve doğru ama sınırda kalan görseller. Hepsi `inspect_*/` klasörüne teşhis içeren adlarla kopyalanır: `HATA__gercek-4_down__rakip-5_down_left__m0.037__xyz.png`
- **Güven eşiği tablosu**: hedef kapsam seviyelerinde eşik, kapsanan accuracy ve reddedilen doğru tahmin sayısı

### Güven kapısı

Model her tahminle birlikte bir güven değeri (seçilen sınıfın cos benzerliği) döndürür. Eşiğin altındaki tahminler `None` olur.

Captcha bağlamında yeniden görüntü almak bedavaya yakın, yanlış yöne gitmek değil. Ölçümler, görsellerin küçük bir kısmını reddetmenin kalan kısımda hatasız çalışmayı sağladığını gösteriyor — ham accuracy'den çok daha kullanışlı bir çalışma noktası.

### TTA (test-time augmentation)

`tta_views=2` yatay çevirmeyi, `4` yatay+dikey kombinasyonlarını ekler. Skorlar ortalanmadan önce yukarıdaki permütasyonlarla orijinal çerçeveye geri taşınır — bu adım olmadan ortalama almak sonucu sessizce bozar. Maliyet view sayısıyla doğrusal artar.

## Sık karşılaşılan sorunlar

**Renk sırası.** Model RGB ile eğitildi (`tf.image.decode_image` RGB üretir). `cv2.imread` BGR döndürür; cv2 kullanıyorsanız `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` ile çevirin. Dosya yolu veya `bytes` verirseniz modül kendisi doğru okur.

**Loss değerinin büyüklüğü kalite ölçütü değildir.** CosFace kaybı `scale` ile ölçeklenir ve margin rampalanırken ilk epoch'larda loss artabilirken accuracy yükselir. Farklı margin/scale ayarlarının loss değerleri karşılaştırılamaz. Bakılacak metrik doğrulama accuracy'si ve sınıf-içi/sınıf-arası mesafe ayrımıdır.

**Blackwell (RTX 50 serisi) GPU'lar.** TensorFlow'un resmî wheel'ları `sm_120` için derlenmiş kernel içermez; PTX'ten JIT derleme yapılır ve ilk çalıştırma uzun sürer. `CUDA_CACHE_MAXSIZE` ayarlamak sonraki çalıştırmaları hızlandırır. CUDA kütüphaneleri yüklenemiyorsa (`Cannot dlopen some GPU libraries`) TensorFlow'un kendi nvidia wheel'larının `lib` klasörlerini `LD_LIBRARY_PATH`'e ekleyin.

## TensorBoard

```bash
tensorboard --logdir=logs_direction
```

## Lisans

Eğitim amaçlı geliştirilmiştir.
