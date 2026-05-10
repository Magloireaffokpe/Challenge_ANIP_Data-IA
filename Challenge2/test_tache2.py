#!/usr/bin/env python3
"""
============================================================
ANIP CHALLENGE — TÂCHE 2 : PRÉDICTION D'ÂGE
Script de test local — Prédiction sur image(s)
============================================================
Usage :
    # Prédire l'âge d'une seule image :
    python test_tache2.py --img photo.jpg

    # Prédire sur un dossier complet :
    python test_tache2.py --dir /chemin/vers/dossier

Modèle attendu :
    ./model_tache2_best.h5 (dans le même dossier que ce script)

Format des noms de fichiers (optionnel, pour afficher l'âge réel) :
    <ID>_<num><Sexe><Age>.jpg   ex: 1234_0M25.jpg
============================================================
"""

import os
import sys
import argparse
import re
import numpy as np
import cv2

# ─────────────────────────────────────────────
# Configuration (identique au notebook)
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model_tache2_best.h5")

IMG_SIZE = (128, 128)  # taille utilisée à l'entraînement
BATCH_SIZE = 32

# Regex pour parser le nom de fichier et extraire l'âge réel
FILENAME_PATTERN = re.compile(r"^\d+_\d+([MF])(\d+)\.jpg$", re.IGNORECASE)


# ─────────────────────────────────────────────
# Imports TensorFlow
# ─────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError


# ─────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────
def load_model(model_path: str):
    if not os.path.exists(model_path):
        sys.exit(f"❌ Modèle introuvable : {model_path}")

    print(f"📂 Chargement du modèle depuis : {model_path}")
    try:
        custom_objects = {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        model = keras_load_model(model_path, custom_objects=custom_objects)
        print("✅ Modèle chargé avec succès.")
        return model
    except Exception as e:
        sys.exit(f"❌ Impossible de charger le modèle : {e}")


# ─────────────────────────────────────────────
# Prétraitement (identique au notebook)
# ─────────────────────────────────────────────
def preprocess_image(path: str) -> np.ndarray:
    """
    Charge et normalise l'image dans [0, 1]
    (même pipeline que le notebook : ImageDataGenerator rescale=1./255).
    """
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"❌ Impossible de lire l'image : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0  # rescale [0, 1]
    return img


def parse_age_from_filename(filename: str):
    """Extrait l'âge réel depuis le nom de fichier si le format correspond."""
    match = FILENAME_PATTERN.match(os.path.basename(filename))
    if match:
        return match.group(1), int(match.group(2))  # (sexe, âge)
    return None, None


# ─────────────────────────────────────────────
# Inférence — une seule image
# ─────────────────────────────────────────────
def predict_single(model, img_path: str) -> dict:
    img = preprocess_image(img_path)
    batch = np.expand_dims(img, axis=0)
    predicted_age = float(model.predict(batch, verbose=0)[0][0])
    predicted_age = max(0.0, predicted_age)  # pas d'âge négatif

    sex, real_age = parse_age_from_filename(img_path)

    result = {
        "filename": os.path.basename(img_path),
        "predicted_age": round(predicted_age, 1),
        "sex": sex,
        "real_age": real_age,
    }
    if real_age is not None:
        result["mae"] = round(abs(predicted_age - real_age), 1)
    return result


# ─────────────────────────────────────────────
# Inférence — dossier d'images
# ─────────────────────────────────────────────
def predict_dir(model, img_dir: str):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])

    if not files:
        print("❌ Aucune image trouvée dans le dossier.")
        return

    print(f"\n🔍 {len(files)} image(s) trouvée(s) dans : {img_dir}\n")

    # En-tête du tableau
    has_real = any(FILENAME_PATTERN.match(f) for f in files)
    if has_real:
        header = (
            f"{'Fichier':<35} {'Sexe':>5} {'Âge réel':>10} {'Prédit':>10} {'MAE':>8}"
        )
    else:
        header = f"{'Fichier':<35} {'Âge prédit':>12}"
    print(header)
    print("─" * len(header))

    results = []
    maes = []

    # Traitement par batch pour la performance
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i : i + BATCH_SIZE]
        imgs = []
        valid_files = []
        for f in batch_files:
            try:
                img = preprocess_image(os.path.join(img_dir, f))
                imgs.append(img)
                valid_files.append(f)
            except SystemExit:
                print(f"  ⚠️ Ignoré : {f}")

        if not imgs:
            continue

        batch = np.stack(imgs, axis=0)
        preds = model.predict(batch, verbose=0).flatten()

        for f, pred in zip(valid_files, preds):
            predicted_age = max(0.0, float(pred))
            sex, real_age = parse_age_from_filename(f)
            row = {
                "filename": f,
                "predicted_age": round(predicted_age, 1),
                "sex": sex,
                "real_age": real_age,
            }
            if real_age is not None:
                row["mae"] = round(abs(predicted_age - real_age), 1)
                maes.append(row["mae"])

            results.append(row)

            if has_real:
                mae_str = (
                    f"{row.get('mae', '-'):>8}"
                    if row.get("mae") is not None
                    else f"{'—':>8}"
                )
                real_str = f"{real_age:>10}" if real_age is not None else f"{'—':>10}"
                sex_str = f"{sex:>5}" if sex is not None else f"{'—':>5}"
                print(f"{f:<35} {sex_str} {real_str} {predicted_age:>10.1f} {mae_str}")
            else:
                print(f"{f:<35} {predicted_age:>12.1f}")

    # Résumé
    print(f"\n📊 Résumé ({len(results)} image(s) traitée(s))")
    ages = [r["predicted_age"] for r in results]
    print(
        f"   Âge prédit — min: {min(ages):.1f}  max: {max(ages):.1f}  moyenne: {np.mean(ages):.1f}"
    )
    if maes:
        print(f"   MAE globale (sur labels disponibles): {np.mean(maes):.2f} ans")

    # Sauvegarde CSV
    import csv

    csv_path = os.path.join(img_dir, "predictions_tache2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "predicted_age", "sex", "real_age", "mae"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Résultats sauvegardés : {csv_path}")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test Tâche 2 — Prédiction d'âge (MobileNetV2)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", help="Chemin vers une image")
    group.add_argument("--dir", help="Dossier d'images à traiter en lot")
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin vers le modèle .h5")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.img:
        res = predict_single(model, args.img)
        print("\n" + "=" * 50)
        print(f"  Fichier       : {res['filename']}")
        print(f"  Âge prédit    : {res['predicted_age']} ans")
        if res["sex"]:
            print(f"  Sexe (label)  : {res['sex']}")
        if res["real_age"] is not None:
            print(f"  Âge réel      : {res['real_age']} ans")
            print(f"  MAE           : {res['mae']} ans")
        print("=" * 50)
    else:
        predict_dir(model, args.dir)


if __name__ == "__main__":
    main()
