#!/usr/bin/env python3
"""
============================================================
ANIP CHALLENGE — TÂCHE 3 : DÉTECTION DE FAUX DOCUMENTS
Script de test local — Prédiction sur image(s)
============================================================
Usage :
    # Tester une seule image :
    python test_tache3.py --img document.jpg

    # Tester un dossier d'images :
    python test_tache3.py --dir /chemin/vers/dossier

    # Préciser le pays (arizona_dl | esp | est | rus) :
    python test_tache3.py --dir /mon/dossier --country esp

Modèle attendu :
    ./final_model.keras (priorité) ou ./mobilenet_v2_checkpoint.h5 (fallback)

Classes :
    0 → Normal (document authentique)
    1 → Forgery (document falsifié)

Pays supportés : arizona_dl, esp, est, rus
============================================================
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2

# ─────────────────────────────────────────────
# Configuration (identique au notebook)
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "final_model.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(SCRIPT_DIR, "mobilenet_v2_checkpoint.h5")

IMG_SIZE = (128, 128)  # taille utilisée à l'entraînement
BATCH_SIZE = 8

COUNTRIES = ["arizona_dl", "esp", "est", "rus"]

CLASS_NAMES = {0: "✅ Normal (authentique)", 1: "🚨 Forgery (falsifié)"}


# ─────────────────────────────────────────────
# Imports TensorFlow
# ─────────────────────────────────────────────
import tensorflow as tf


# ─────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────
def load_model(model_path: str):
    if not os.path.exists(model_path):
        sys.exit(
            f"❌ Modèle introuvable : {model_path}\n"
            f"   Vérifiez que le fichier est bien dans : {SCRIPT_DIR}\n"
            f"   (final_model.keras ou mobilenet_v2_checkpoint.h5)"
        )

    print(f"📂 Chargement du modèle depuis : {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print("✅ Modèle chargé avec succès.")
        print(f"   Sortie : {model.output_shape}  (2 classes : normal / forgery)")
        return model
    except Exception as e:
        sys.exit(f"❌ Impossible de charger le modèle : {e}")


# ─────────────────────────────────────────────
# Prétraitement (identique au notebook)
# ─────────────────────────────────────────────
def preprocess_image(path: str) -> np.ndarray:
    """
    Charge et normalise dans [0, 1]
    (même pipeline que decode_img du notebook : convert_image_dtype → [0,1]).
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0  # [0, 1]
    return img


# ─────────────────────────────────────────────
# Inférence — une seule image
# ─────────────────────────────────────────────
def predict_single(model, img_path: str, country: str = None) -> dict:
    img = preprocess_image(img_path)
    batch = np.expand_dims(img, axis=0)
    probs = model.predict(batch, verbose=0)[0]  # shape (2,)

    pred_label = int(np.argmax(probs))
    confidence = float(np.max(probs))
    prob_normal = float(probs[0])
    prob_forgery = float(probs[1])

    return {
        "filename": os.path.basename(img_path),
        "country": country or "—",
        "prediction": pred_label,
        "label": CLASS_NAMES[pred_label],
        "confidence": round(confidence * 100, 1),
        "prob_normal": round(prob_normal * 100, 1),
        "prob_forgery": round(prob_forgery * 100, 1),
    }


# ─────────────────────────────────────────────
# Inférence — dossier d'images
# ─────────────────────────────────────────────
def _collect_images(img_dir: str, country: str = None):
    """
    Collecte les images depuis un dossier.
    Si country est précisé, cherche aussi dans les sous-dossiers de pays.
    """
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    rows = []

    # Sous-dossiers par pays (structure du notebook : test/<country>/*.img)
    if country:
        targets = [country]
    else:
        # Chercher si des sous-dossiers de pays existent
        sub = [
            d
            for d in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, d)) and d in COUNTRIES
        ]
        targets = sub if sub else []

    if targets:
        for c in targets:
            c_dir = os.path.join(img_dir, c)
            if not os.path.exists(c_dir):
                print(f"⚠️  Dossier pays introuvable : {c_dir}")
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for f in glob.glob(os.path.join(c_dir, ext)):
                    rows.append(
                        {"filepath": f, "country": c, "filename": os.path.basename(f)}
                    )
    else:
        # Dossier plat
        for f in os.listdir(img_dir):
            if f.lower().endswith(valid_ext):
                rows.append(
                    {
                        "filepath": os.path.join(img_dir, f),
                        "country": "—",
                        "filename": f,
                    }
                )

    return sorted(rows, key=lambda r: (r["country"], r["filename"]))


def predict_dir(model, img_dir: str, country: str = None):
    rows = _collect_images(img_dir, country)
    if not rows:
        print("❌ Aucune image trouvée.")
        return

    print(f"\n🔍 {len(rows)} image(s) trouvée(s) dans : {img_dir}\n")
    header = f"{'Fichier':<35} {'Pays':<12} {'Résultat':<25} {'Confiance':>10} {'P(normal)':>10} {'P(faux)':>10}"
    print(header)
    print("─" * len(header))

    results = []
    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i : i + BATCH_SIZE]
        imgs = []
        valid_rows = []

        for row in batch_rows:
            try:
                img = preprocess_image(row["filepath"])
                imgs.append(img)
                valid_rows.append(row)
            except ValueError as e:
                print(f"  ⚠️ {e}")

        if not imgs:
            continue

        batch = np.stack(imgs, axis=0)
        all_probs = model.predict(batch, verbose=0)  # (N, 2)

        for row, probs in zip(valid_rows, all_probs):
            pred_label = int(np.argmax(probs))
            confidence = float(np.max(probs))
            res = {
                "filename": row["filename"],
                "country": row["country"],
                "prediction": pred_label,
                "label": CLASS_NAMES[pred_label],
                "confidence": round(confidence * 100, 1),
                "prob_normal": round(float(probs[0]) * 100, 1),
                "prob_forgery": round(float(probs[1]) * 100, 1),
            }
            results.append(res)
            print(
                f"{row['filename']:<35} {row['country']:<12} {res['label']:<25} "
                f"{res['confidence']:>9.1f}% {res['prob_normal']:>9.1f}% {res['prob_forgery']:>9.1f}%"
            )

    # Résumé global
    normal_count = sum(1 for r in results if r["prediction"] == 0)
    forgery_count = sum(1 for r in results if r["prediction"] == 1)
    print(f"\n📊 Résumé ({len(results)} image(s))")
    print(f"   ✅ Authentiques : {normal_count}")
    print(f"   🚨 Falsifiés    : {forgery_count}")

    # Résumé par pays
    countries_seen = sorted(set(r["country"] for r in results))
    if len(countries_seen) > 1:
        print("\n   Par pays :")
        for c in countries_seen:
            c_res = [r for r in results if r["country"] == c]
            c_ok = sum(1 for r in c_res if r["prediction"] == 0)
            c_bad = sum(1 for r in c_res if r["prediction"] == 1)
            print(f"     {c:<14}  ✅ {c_ok}  🚨 {c_bad}  (total {len(c_res)})")

    # Sauvegarde CSV
    import csv

    csv_path = os.path.join(img_dir, "predictions_tache3.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "country",
            "prediction",
            "confidence",
            "prob_normal",
            "prob_forgery",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n💾 Résultats sauvegardés : {csv_path}")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test Tâche 3 — Détection de faux documents (MobileNetV2)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", help="Chemin vers une image à analyser")
    group.add_argument(
        "--dir", help="Dossier d'images (peut contenir des sous-dossiers pays)"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH, help="Chemin vers le modèle (.keras ou .h5)"
    )
    parser.add_argument(
        "--country",
        choices=COUNTRIES,
        default=None,
        help="Pays cible pour filtrer dans un dossier (arizona_dl | esp | est | rus)",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.img:
        res = predict_single(model, args.img, args.country)
        print("\n" + "=" * 55)
        print(f"  Fichier        : {res['filename']}")
        print(f"  Pays           : {res['country']}")
        print(f"  Résultat       : {res['label']}")
        print(f"  Confiance      : {res['confidence']}%")
        print(f"  P(authentique) : {res['prob_normal']}%")
        print(f"  P(falsifié)    : {res['prob_forgery']}%")
        print("=" * 55)
    else:
        predict_dir(model, args.dir, args.country)


if __name__ == "__main__":
    main()
