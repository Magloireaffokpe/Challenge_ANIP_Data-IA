#!/usr/bin/env python3
"""
============================================================
ANIP CHALLENGE — TÂCHE 1 : RECONNAISSANCE FACIALE (Siamese)
Script de test local — Prédiction sur images
============================================================
Usage :
    # Vérifier si 2 images sont la même personne :
    python test_tache1.py --img1 photo1.jpg --img2 photo2.jpg

    # Tester un dossier de paires (format: ID_0.jpg / ID_1.jpg) :
    python test_tache1.py --pairs_dir /chemin/vers/dossier

Modèle attendu :
    ./best_siamese_model.keras (dans le même dossier que ce script)
============================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2

# ─────────────────────────────────────────────
# Configuration (identique au notebook)
# ─────────────────────────────────────────────
# Chemin relatif au script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_siamese_model.keras")

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Embedding dim et seuil (notebook CONFIG)
EMBEDDING_DIM = 256
DROPOUT_RATE = 0.3
L2_REG = 1e-4
DISTANCE_THRESHOLD = 0.7  # seuil contrastive (distance ≤ seuil → même personne)
MARGIN = 1.0


# ─────────────────────────────────────────────
# Imports TensorFlow (après la config)
# ─────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


# ─────────────────────────────────────────────
# Architecture Siamese (identique au notebook)
# ─────────────────────────────────────────────
def build_embedding_model(
    img_h=IMG_HEIGHT,
    img_w=IMG_WIDTH,
    embedding_dim=EMBEDDING_DIM,
    dropout_rate=DROPOUT_RATE,
    l2_reg=L2_REG,
):
    base_model = MobileNetV2(
        weights=None,  # poids chargés depuis le fichier .keras
        include_top=False,
        input_shape=(img_h, img_w, 3),
    )
    inputs = keras.Input(shape=(img_h, img_w, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        512, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(embedding_dim, kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    embeddings = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1), name="l2_normalize"
    )(x)
    return Model(inputs, embeddings, name="embedding_model"), base_model


def build_siamese_model(embedding_model, img_h=IMG_HEIGHT, img_w=IMG_WIDTH):
    input_1 = keras.Input(shape=(img_h, img_w, 3))
    input_2 = keras.Input(shape=(img_h, img_w, 3))
    emb1 = embedding_model(input_1)
    emb2 = embedding_model(input_2)
    distance = layers.Lambda(
        lambda v: tf.reduce_sum(tf.square(v[0] - v[1]), axis=1, keepdims=True),
        name="euclidean_distance",
    )([emb1, emb2])
    return Model([input_1, input_2], distance, name="siamese_model")


# ─────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────
def load_model(model_path: str):
    if not os.path.exists(model_path):
        sys.exit(f"❌ Modèle introuvable : {model_path}")

    print(f"📂 Chargement du modèle depuis : {model_path}")

    # Méthode 1 : recréer l'architecture puis injecter les poids
    try:
        embedding_model, _ = build_embedding_model()
        siamese = build_siamese_model(embedding_model)

        def contrastive_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            return tf.reduce_mean(
                y_true * y_pred
                + (1 - y_true)
                * tf.square(tf.maximum(MARGIN - tf.sqrt(y_pred + 1e-6), 0))
            )

        siamese.compile(optimizer="adam", loss=contrastive_loss)
        siamese.load_weights(model_path)
        print("✅ Modèle chargé (méthode poids).")
        return siamese

    except Exception as e1:
        print(f"⚠️ Méthode 1 échouée ({e1}), tentative méthode 2…")

    # Méthode 2 : load_model avec custom_objects
    try:

        def l2_normalize(x):
            return tf.nn.l2_normalize(x, axis=1)

        custom_objects = {
            "l2_normalize": l2_normalize,
            "euclidean_distance": lambda v: tf.reduce_sum(
                tf.square(v[0] - v[1]), axis=1, keepdims=True
            ),
        }
        model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
        print("✅ Modèle chargé (méthode load_model).")
        return model

    except Exception as e2:
        sys.exit(
            f"❌ Impossible de charger le modèle.\n  Erreur 1: {e1}\n  Erreur 2: {e2}"
        )


# ─────────────────────────────────────────────
# Prétraitement image (identique au notebook)
# ─────────────────────────────────────────────
def preprocess_image(path: str) -> np.ndarray:
    """
    Charge, redimensionne et normalise une image dans [-1, 1]
    (même pipeline que la fonction load_and_preprocess_image du notebook).
    """
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"❌ Impossible de lire l'image : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 127.5 - 1.0  # normalisation [-1, 1]
    return img


# ─────────────────────────────────────────────
# Inférence sur une paire
# ─────────────────────────────────────────────
def predict_pair(
    model, img1_path: str, img2_path: str, threshold: float = DISTANCE_THRESHOLD
):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    batch1 = np.expand_dims(img1, axis=0)
    batch2 = np.expand_dims(img2, axis=0)

    distance = float(model.predict([batch1, batch2], verbose=0)[0][0])
    same_person = distance <= threshold

    return {
        "image1": os.path.basename(img1_path),
        "image2": os.path.basename(img2_path),
        "distance": distance,
        "threshold": threshold,
        "prediction": 1 if same_person else 0,
        "label": "✅ Même personne" if same_person else "❌ Personnes différentes",
    }


# ─────────────────────────────────────────────
# Mode dossier de paires (ID_0.jpg / ID_1.jpg)
# ─────────────────────────────────────────────
def predict_pairs_dir(model, pairs_dir: str, threshold: float = DISTANCE_THRESHOLD):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(pairs_dir) if f.lower().endswith(valid_ext)]

    groups = {}
    for f in files:
        prefix = f.split("_")[0]
        groups.setdefault(prefix, []).append(f)

    pairs = []
    for prefix, flist in sorted(groups.items()):
        if len(flist) == 2:
            flist.sort()
            pairs.append((prefix, flist[0], flist[1]))
        else:
            print(f"⚠️  Préfixe '{prefix}' ignoré : {len(flist)} image(s) trouvée(s).")

    if not pairs:
        print("❌ Aucune paire valide trouvée.")
        return

    print(f"\n🔍 {len(pairs)} paire(s) détectée(s) dans : {pairs_dir}\n")
    print(f"{'ID':<10} {'Image 1':<30} {'Image 2':<30} {'Distance':>10} {'Résultat'}")
    print("─" * 100)

    results = []
    for prefix, f1, f2 in pairs:
        res = predict_pair(
            model, os.path.join(pairs_dir, f1), os.path.join(pairs_dir, f2), threshold
        )
        results.append({"pair_id": prefix, **res})
        print(
            f"{prefix:<10} {f1:<30} {f2:<30} {res['distance']:>10.4f}   {res['label']}"
        )

    same = sum(1 for r in results if r["prediction"] == 1)
    print(
        f"\n📊 Résumé : {same} même(s) personne(s) / {len(results) - same} personne(s) différente(s)"
    )

    # Sauvegarde CSV
    import csv

    csv_path = os.path.join(pairs_dir, "predictions_tache1.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pair_id", "prediction", "distance", "threshold"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "pair_id": r["pair_id"],
                    "prediction": r["prediction"],
                    "distance": f"{r['distance']:.6f}",
                    "threshold": r["threshold"],
                }
            )
    print(f"💾 Résultats sauvegardés : {csv_path}")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test Tâche 1 — Reconnaissance faciale (réseau Siamese)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img1", help="Chemin vers la première image")
    group.add_argument(
        "--pairs_dir", help="Dossier contenant des paires (ID_0.jpg / ID_1.jpg)"
    )
    parser.add_argument(
        "--img2", help="Chemin vers la seconde image (requis avec --img1)"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH, help="Chemin vers le modèle .keras"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DISTANCE_THRESHOLD,
        help=f"Seuil de distance (défaut: {DISTANCE_THRESHOLD})",
    )
    args = parser.parse_args()

    if args.img1 and not args.img2:
        parser.error("--img2 est requis quand --img1 est utilisé.")

    model = load_model(args.model)

    if args.img1:
        res = predict_pair(model, args.img1, args.img2, args.threshold)
        print("\n" + "=" * 50)
        print(f"  Image 1   : {res['image1']}")
        print(f"  Image 2   : {res['image2']}")
        print(f"  Distance  : {res['distance']:.4f}  (seuil = {res['threshold']})")
        print(f"  Résultat  : {res['label']}")
        print("=" * 50)
    else:
        predict_pairs_dir(model, args.pairs_dir, args.threshold)


if __name__ == "__main__":
    main()
