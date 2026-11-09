"""
Keras (.keras) modelini ONNX (.onnx) formatına dönüştürme betiği
Kullanım:
    python export_onnx.py --meta direction_embedding_v8_meta.json
"""

import argparse
import json
import os
import sys

def convert_to_onnx(meta_path, output_path=None):
    base_dir = os.path.dirname(os.path.abspath(meta_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    keras_rel = meta["embedding_model"]
    keras_path = keras_rel if os.path.isabs(keras_rel) else os.path.join(base_dir, keras_rel)
    
    if output_path is None:
        name_without_ext = os.path.splitext(keras_path)[0]
        output_path = name_without_ext + ".onnx"

    print(f"[*] Keras modeli yükleniyor: {keras_path}")
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(keras_path)
    img_size = tuple(meta.get("img_size", [128, 128]))
    
    input_signature = (
        tf.TensorSpec(
            shape=(None, img_size[0], img_size[1], 3),
            dtype=tf.float32,
            name="image"
        ),
    )

    print(f"[*] ONNX formatına dönüştürülüyor -> {output_path}")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        output_path=output_path,
        opset=13
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Başarılı! ONNX dosya boyutu: {size_mb:.2f} MB")
    
    # İsteğe bağlı meta_onnx.json üret
    meta_onnx = dict(meta)
    meta_onnx["embedding_model"] = os.path.basename(output_path)
    meta_onnx["runtime"] = "onnxruntime"
    
    meta_onnx_path = os.path.splitext(meta_path)[0] + "_onnx.json"
    with open(meta_onnx_path, "w", encoding="utf-8") as f:
        json.dump(meta_onnx, f, indent=2)
    print(f"[+] ONNX metadata dosyası kaydedildi: {meta_onnx_path}")
    return output_path, meta_onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keras Captcha Modelini ONNX'e dönüştür")
    parser.add_argument("--meta", default="direction_embedding_v8_meta.json", help="Meta JSON dosya yolu")
    parser.add_argument("--out", default=None, help="Çıktı .onnx dosya yolu (opsiyonel)")
    args = parser.parse_args()
    
    convert_to_onnx(args.meta, args.out)
