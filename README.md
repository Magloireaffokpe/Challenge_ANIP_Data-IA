# 🧪 ANIP Challenge – Scripts de test

Ce dépôt contient les scripts de test pour les trois tâches du challenge ANIP BENIN :

- **Tâche 1** : Reconnaissance faciale (réseau Siamese)
- **Tâche 2** : Prédiction d’âge (régression)
- **Tâche 3** : Détection de faux documents (classification binaire)

Les scripts sont autonomes et portables : ils chargent les modèles depuis leur propre dossier grâce à des chemins relatifs.

---

## 📦 Prérequis

```bash
pip install tensorflow opencv-python numpy scikit-learn

🚀 Utilisation

# ─────────────────────────────────────────────
# Tâche 1 – Reconnaissance faciale
# ─────────────────────────────────────────────


cd Challenge1

# Comparer deux images
python test_tache1.py --img1 tache1_0.jpg --img2 tache1_1.jpg

# Seuil personnalisé (défaut 0.7)
python test_tache1.py --img1 a.jpg --img2 b.jpg --threshold 0.5

# Dossier contenant des paires (nommage : ID_0.jpg / ID_1.jpg)
python test_tache1.py --pairs_dir /chemin/vers/mon_dossier

Résultat : distance euclidienne + décision.
Un fichier predictions_tache1.csv est créé dans le dossier traité.


# ─────────────────────────────────────────────
# Tâche 2 – Prédiction d’âge
# ─────────────────────────────────────────────

cd Challenge2

# Prédire sur une image
python test_tache2.py --img tache2.jpg

# Traiter un dossier entier
python test_tache2.py --dir /chemin/vers/dossier

# Avec un modèle personnalisé
python test_tache2.py --img photo.jpg --model /autre/modele.h5

Extraction automatique de l’âge réel (si le fichier respecte le format <ID>_<num><M|F><age>.jpg, ex: 1234_0M25.jpg).
La MAE est calculée et un CSV predictions_tache2.csv est généré.


# ─────────────────────────────────────────────
# Tâche 3 – Détection de faux documents
# ─────────────────────────────────────────────


cd Challenge3

# Analyser une image
python test_tache3.py --img tache3_esp.png

# Dossier avec sous-dossiers par pays (arizona_dl, esp, est, rus)
python test_tache3.py --dir /chemin/vers/dossier

# Forcer un pays spécifique
python test_tache3.py --dir /chemin/vers/dossier --country esp

Sortie : classe prédite (authentique / falsifié) + probabilités.
Un CSV predictions_tache3.csv est enregistré dans le dossier analysé.

📁 Arborescence du projet (à placer dans votre dépôt)

Challenge1/
├── best_siamese_model.keras
├── Notebook_Tache_1.ipynb
├── tache1_0.jpg
├── tache1_1.jpg
├── tache1_2.jpg
├── tache1_3.jpg
└── test_tache1.py
Challenge2/
├── model_tache2_best.h5
├── Notebook_Tache_2.ipynb
├── tache2.jpg
└── test_tache2.py
Challenge3/
├── final_model.keras
├── Notebook_Tache_3.ipynb
├── tache3_arizona.png
├── tache3_esp.png
└── test_tache3.py
README.md
