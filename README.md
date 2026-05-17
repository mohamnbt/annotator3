# COSMER Annotator — Documentation complète

> **Outil d'annotation et d'apprentissage automatique pour l'estimation de la vitesse de courant marin**  
> par analyse visuelle de câbles de mouillage sous-marins.  
> Développé au **Laboratoire COSMER — Université de Toulon**

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Architecture générale](#2-architecture-générale)
3. [Prérequis](#3-prérequis)
4. [Installation](#4-installation)
5. [Démarrage de l'application](#5-démarrage-de-lapplication)
6. [Guide d'utilisation pas à pas](#6-guide-dutilisation-pas-à-pas)
7. [Calculs effectués dans le backend](#7-calculs-effectués-dans-le-backend)
8. [API REST — référence complète](#8-api-rest--référence-complète)
9. [Structure des fichiers de données](#9-structure-des-fichiers-de-données)
10. [Entraînement IA et modèles](#10-entraînement-ia-et-modèles)
11. [Auto-annotation YOLO](#11-auto-annotation-yolo)
12. [Export des données](#12-export-des-données)
13. [Étendre / modifier l'application](#13-étendre--modifier-lapplication)
14. [FAQ et dépannage](#14-faq-et-dépannage)

---

## 1. Contexte et objectif

Lorsqu'un câble de mouillage est soumis à un courant marin, il se défléchit et forme un angle par rapport à la verticale. Cet angle est directement corrélé à la vitesse du courant. L'application **COSMER Annotator** a été créée pour :

- **Annoter** des images de câbles de mouillage (issues de caméras sous-marines ou de drones DJI) en traçant leur ligne centrale (centerline) ou leur contour.
- **Calculer automatiquement** l'angle de déflexion du câble, l'angle accordal et un indice de courbure.
- **Construire un dataset** au format YOLO pour l'entraînement de modèles de segmentation.
- **Entraîner un réseau de neurones** (ResNet-18) pour prédire la vitesse de courant (Vc) directement à partir d'une image.
- **Visualiser** les statistiques du dataset et les courbes Vc = f(θ).

L'application est conçue pour être utilisée par des chercheurs sans compétences en développement logiciel : l'interface graphique guide l'utilisateur de bout en bout.

---

## 2. Architecture générale

```
annotator3/
├── backend/
│   ├── main.py          ← API FastAPI (Python) — toute la logique métier
│   ├── requirements.txt ← dépendances Python
│   └── best.pt          ← (optionnel) modèle YOLO pré-entraîné
├── frontend/
│   ├── src/             ← Interface React + TypeScript (Vite)
│   └── package.json
├── data/
│   ├── sessions/        ← données générées (images, annotations, labels)
│   └── models/          ← modèles PyTorch (.pth) entraînés
├── start.sh             ← script de démarrage tout-en-un
└── requirements.txt     ← dépendances Python (racine)
```

**Flux de données :**

```
Navigateur (React) ──HTTP/REST──► FastAPI (Python)
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼             ▼
                     data/sessions  OpenCV/NumPy  PyTorch/YOLO
                     (JSON + images) (calculs)   (modèles IA)
```

Le **backend** tourne sur `http://localhost:8000` et expose une API REST.  
Le **frontend** tourne sur `http://localhost:5173` et communique exclusivement via cette API.

---

## 3. Prérequis

### Système

| Outil | Version minimale | Usage |
|-------|-----------------|-------|
| Python | 3.10+ | Backend FastAPI |
| Node.js | 18+ | Frontend React/Vite |
| npm | 9+ | Gestion des paquets frontend |
| FFmpeg | toute version récente | Extraction de frames vidéo |

### Installation de FFmpeg

```bash
# macOS (avec Homebrew)
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Télécharger depuis https://ffmpeg.org/download.html et ajouter au PATH
```

---

## 4. Installation

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/mohamnbt/annotator3.git
cd annotator3
```

### Étape 2 — Installer les dépendances Python

```bash
# Depuis la racine du projet
pip install -r requirements.txt

# Ou depuis le dossier backend
pip install -r backend/requirements.txt
```

Dépendances Python installées :

| Paquet | Rôle |
|--------|------|
| `fastapi` | Framework web API |
| `uvicorn` | Serveur ASGI |
| `opencv-python-headless` | Traitement d'images (centerline, masques) |
| `python-multipart` | Upload de fichiers |
| `pydantic` | Validation des données |
| `ultralytics` | Auto-annotation YOLO (optionnel) |

### Étape 3 — Installer les dépendances frontend

```bash
cd frontend
npm install
cd ..
```

### Étape 4 (optionnel) — IA : PyTorch + YOLO

Pour pouvoir **entraîner un modèle de régression Vc** et utiliser **l'auto-annotation YOLO** :

```bash
# PyTorch (choisir selon votre matériel)
pip install torch torchvision          # CPU / CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU seul

# YOLO (déjà inclus dans requirements, mais pour être explicite)
pip install ultralytics

# Placer un modèle YOLO pré-entraîné (si disponible)
cp votre_modele.pt backend/best.pt
```

> Sans PyTorch, toutes les fonctions d'annotation manuelle, d'export et de statistiques restent disponibles. Seuls l'entraînement et la prédiction Vc sont désactivés.

---

## 5. Démarrage de l'application

### Méthode rapide (recommandée)

```bash
chmod +x start.sh
./start.sh
```

Ce script :
1. Démarre le backend FastAPI sur le port **8000**
2. Démarre le frontend Vite sur le port **5173**
3. Gère l'arrêt propre des deux processus avec `Ctrl+C`

**Ouvrir dans le navigateur :** `http://localhost:5173`

### Méthode manuelle (deux terminaux)

**Terminal 1 — Backend :**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend :**
```bash
cd frontend
npm run dev
```

### Vérification

- API backend : `http://localhost:8000/docs` → interface Swagger interactive
- Frontend : `http://localhost:5173`

---

## 6. Guide d'utilisation pas à pas

### 6.1 Créer une session

Une **session** est un ensemble d'images liées à une même expérience (même câble, même conditions générales).

1. Cliquer sur **« Nouvelle session »**
2. Saisir un nom (sera automatiquement nettoyé : espaces → `_`, accents supprimés)
3. Optionnel : ajouter une description et un dossier de classement
4. Valider → la session apparaît dans la liste

### 6.2 Importer des images

**Option A — Upload direct :**  
Glisser-déposer des fichiers `.jpg`, `.jpeg` ou `.png` dans la session.

**Option B — Extraction depuis une vidéo :**  
1. Cliquer sur **« Importer une vidéo »**
2. Choisir le fichier vidéo (DJI, GoPro, etc.)
3. Régler l'intervalle d'extraction (par défaut : 1 frame toutes les 240 frames ≈ 1 frame/8s à 30fps)
4. L'extraction se fait **en arrière-plan** via FFmpeg — une barre de progression s'affiche
5. Les frames extraites apparaissent automatiquement dans la session

### 6.3 Annoter une image

1. Cliquer sur une image dans la liste (statut : **"À annoter"**)
2. L'éditeur s'ouvre avec l'image à pleine taille
3. Renseigner les **conditions expérimentales** :
   - `current_speed_cm_s` — vitesse de courant mesurée (cm/s) ← **champ clé pour l'IA**
   - `current_direction` — direction du courant
   - `camera_angle` — angle de la caméra
   - `wave_amplitude_cm`, `wave_length_cm`, `wave_speed_cm_s` — paramètres de houle
   - `water_depth_m`, `cable_tension_n` — paramètres physiques
   - `annotator_name` — nom de l'annotateur
   - `notes` — remarques libres

4. Tracer l'annotation :
   - **Mode centerline** : cliquer le long de l'axe central du câble, du haut vers le bas
   - **Mode contour** : tracer les bords gauche et droit du câble séparément

5. Cliquer **« Enregistrer »** → l'image passe au statut **"Annotée"**

> **Auto-annotation YOLO** : si un fichier `backend/best.pt` est présent, cliquer sur **« Prédire »** pour obtenir une annotation automatique à corriger manuellement.

### 6.4 Naviguer entre images

- Flèches **← →** du clavier ou boutons de navigation
- Le statut de chaque image est visible dans la liste (À annoter / Annotée / Ignorée)
- Une image peut être marquée **"Ignorée"** si elle est inutilisable

### 6.5 Consulter les statistiques

L'onglet **Statistiques** de la session affiche :
- Taux d'annotation (annotées / ignorées / restantes)
- Histogramme des vitesses de courant
- Histogramme des angles de câble
- Histogramme de l'indice de courbure
- Répartition par direction de courant et angle de caméra
- Nuage de points amplitude/vitesse
- Alertes de déséquilibre du dataset
- Graphique Vc = f(θ) (vitesse en fonction de l'angle)

### 6.6 Exporter le dataset

L'export génère un fichier `.zip` contenant :
- `images/train/` et `images/val/` — images réparties aléatoirement (80/20)
- `labels/train/` et `labels/val/` — fichiers `.txt` au format YOLO
- `annotations/train/` et `annotations/val/` — fichiers `.json` complets
- `dataset.yaml` — fichier de configuration YOLO
- `annotations.csv` — tableau récapitulatif de toutes les annotations

Un **export global** (toutes sessions confondues) est également disponible depuis l'onglet global.

---

## 7. Calculs effectués dans le backend

### 7.1 Angle du câble — `calc_cable_angle(points)`

Cette fonction prend en entrée une liste de points `[{x, y}, ...]` représentant la centerline du câble et retourne trois valeurs :

#### Angle de régression (`cable_angle_deg`)

La direction principale du câble est calculée par **Analyse en Composantes Principales (ACP)** sur les coordonnées des points :

1. Calcul de la matrice de covariance des coordonnées centrées
2. Décomposition en valeurs propres → le vecteur propre associé à la plus grande valeur propre donne la direction principale
3. L'angle est calculé comme : `θ = arctan(|dx| / |dy|)` en degrés

Cet angle représente l'inclinaison du câble par rapport à la **verticale** (0° = câble parfaitement vertical, 90° = horizontal).

#### Angle accordal (`cable_angle_chord_deg`)

Angle de la corde reliant le **premier et le dernier point** de la centerline :

```
θ_chord = arctan(|x_last - x_first| / |y_first - y_last|)
```

#### Indice de courbure (`cable_curvature_index`)

Différence absolue entre l'angle de régression et l'angle accordal :

```
ICourbure = |θ_reg - θ_chord|
```

Un indice élevé indique un câble fortement courbé (non rectiligne), ce qui peut indiquer une interaction avec la houle.

### 7.2 Rééchantillonnage équidistant — `_resample_equidistant(points, n)`

Pour garantir la **cohérence du dataset**, tous les câbles d'une même session sont représentés par le **même nombre de points équidistants** :

1. Calcul des longueurs des segments entre points consécutifs
2. Calcul des abscisses curvilignes cumulées
3. Interpolation linéaire pour placer `n` points uniformément espacés le long de la courbe

Le nombre cible `n` est déterminé automatiquement : c'est le **minimum** entre le nombre de points de la nouvelle annotation et le nombre de points de toutes les annotations existantes. Ainsi, si une session contient des annotations à 40 points et qu'une nouvelle annotation est faite avec 30 points, toutes les annotations existantes sont **rééchantillonnées rétroactivement** à 30 points.

### 7.3 Extraction de la centerline depuis un masque YOLO — `extract_centerline_from_mask`

Quand le modèle YOLO détecte un câble, il retourne un **masque polygonal**. La centerline est extraite ainsi :

1. Le masque polygonal est rasterisé en image binaire
2. La **squelettisation morphologique** (`skimage.morphology.skeletonize`) réduit le masque à une ligne centrale de 1 pixel d'épaisseur
3. Les pixels du squelette sont ordonnés par projection sur l'axe principal (ACP)
4. 40 points équidistants sont sélectionnés le long du squelette

Si la squelettisation échoue, une méthode de secours par **tranches horizontales ou verticales** calcule le centre de masse du masque à intervalles réguliers.

### 7.4 Normalisation YOLO — `write_yolo_label`

Les coordonnées des points (en pixels) sont converties en coordonnées normalisées `[0, 1]` :

```
x_norm = x_pixel / image_width
y_norm = y_pixel / image_height
```

Le fichier `.txt` résultant contient une seule ligne : `0 x1 y1 x2 y2 ... xn yn` (classe 0 = câble).

### 7.5 Modèle de régression Vc — ResNet-18

Le modèle d'estimation de la vitesse de courant est un **ResNet-18** modifié :

```
ResNet-18 (backbone) → FC(512→128) → ReLU → Dropout(0.3) → FC(128→1)
```

- **Entrée** : image RGB 224×224 pixels, normalisée (ImageNet mean/std)
- **Sortie** : valeur scalaire = vitesse estimée en cm/s
- **Fonction de perte** : MSE (Mean Squared Error)
- **Optimiseur** : Adam (lr=1e-3)
- **Scheduler** : StepLR (réduction du lr par 0.5 tous les `epochs/3` epochs)
- **Augmentation** : flip horizontal aléatoire, variation de luminosité/contraste
- **Split** : 80% entraînement / 20% validation (aléatoire)
- **Matériel** : détection automatique GPU CUDA, Apple Silicon MPS, ou CPU

Les métriques calculées à la fin de l'entraînement :
- **MAE** (Mean Absolute Error) en cm/s
- **RMSE** (Root Mean Squared Error) en cm/s
- Courbes de loss train/val par epoch
- Courbe Vc_annotations = f(θ) — données réelles
- Courbe Vc_NN = f(θ) — prédictions du modèle

---

## 8. API REST — référence complète

L'API complète est consultable via Swagger : `http://localhost:8000/docs`

### Sessions

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/sessions` | Liste toutes les sessions |
| `POST` | `/api/sessions` | Crée une session (`name`, `description`, `folder`) |
| `GET` | `/api/sessions/{name}` | Détails d'une session |
| `PATCH` | `/api/sessions/{name}` | Modifie description/dossier |
| `DELETE` | `/api/sessions/{name}` | Supprime une session et ses données |
| `POST` | `/api/sessions/batch-move` | Déplace plusieurs sessions dans un dossier |

### Images

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/sessions/{name}/images` | Upload d'images |
| `GET` | `/api/sessions/{name}/images/{filename}` | Récupère une image |
| `DELETE` | `/api/sessions/{name}/images/{filename}` | Supprime une image + annotation + label |
| `POST` | `/api/sessions/{name}/images/{filename}/ignore` | Marque l'image comme ignorée |
| `GET` | `/api/sessions/{name}/images/{filename}/predict` | Auto-annotation YOLO |

### Vidéo

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/sessions/{name}/video` | Upload vidéo + extraction frames (FFmpeg) |
| `GET` | `/api/sessions/{name}/video/progress` | Progression de l'extraction |

### Annotations

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/sessions/{name}/annotations/{stem}` | Récupère une annotation |
| `POST` | `/api/sessions/{name}/annotations/{stem}` | Sauvegarde une annotation |
| `GET` | `/api/sessions/{name}/last-conditions` | Récupère les dernières conditions saisies |

### Statistiques et export

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/sessions/{name}/statistics` | Statistiques complètes de la session |
| `GET` | `/api/sessions/{name}/export/stats` | Aperçu du split train/val |
| `GET` | `/api/sessions/{name}/export/download` | Télécharge le dataset ZIP |
| `GET` | `/api/export/global/download` | Export ZIP global (toutes sessions) |
| `GET` | `/api/sessions/{name}/angle-vc-data` | Points θ/Vc pour graphique |

### Entraînement IA

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/sessions/{name}/train` | Lance l'entraînement (session) |
| `GET` | `/api/sessions/{name}/train/progress` | Progression de l'entraînement |
| `POST` | `/api/train/global` | Lance l'entraînement global |
| `GET` | `/api/train/global/progress` | Progression entraînement global |
| `GET` | `/api/train/global/angle-vc-data` | Points θ/Vc toutes sessions |
| `GET` | `/api/models` | Liste les modèles entraînés |
| `GET` | `/api/models/{filename}/download` | Télécharge un modèle `.pth` |
| `POST` | `/api/predict` | Prédit Vc sur une image |

---

## 9. Structure des fichiers de données

```
data/
├── sessions/
│   └── {session_name}/
│       ├── session.json          ← métadonnées de la session
│       ├── images/
│       │   ├── frame_000001.jpg
│       │   └── ...
│       ├── annotations/
│       │   ├── frame_000001.json ← annotation complète par image
│       │   └── ...
│       └── labels/
│           ├── frame_000001.txt  ← label YOLO (format polygone)
│           └── ...
└── models/
    ├── {session_name}_vc_model.pth  ← modèle entraîné par session
    └── global_vc_model.pth          ← modèle entraîné toutes sessions
```

### Format `session.json`

```json
{
  "name": "session_courant_10cms",
  "description": "Expérience bassin, courant 10 cm/s",
  "folder": "experiences_2025",
  "created_at": "2025-06-01T10:00:00",
  "images": [
    {
      "filename": "frame_000001.jpg",
      "status": "annotated",
      "added_at": "2025-06-01T10:05:00",
      "source_video": "video_dji.mp4"
    }
  ]
}
```

Statuts possibles d'une image : `"to_annotate"` | `"annotated"` | `"ignored"`

### Format `{stem}.json` (annotation)

```json
{
  "points": [
    {"x": 512.3, "y": 45.1},
    {"x": 514.7, "y": 89.3},
    ...
  ],
  "n_points_normalized": 40,
  "image_width": 1920,
  "image_height": 1080,
  "annotation_mode": "centerline",
  "cable_angle_deg": 12.453,
  "cable_angle_chord_deg": 12.107,
  "cable_curvature_index": 0.346,
  "conditions": {
    "current_speed_cm_s": 10.5,
    "current_direction": "est",
    "camera_angle": "90",
    "wave_amplitude_cm": 2.0,
    "wave_length_cm": 50.0,
    "wave_speed_cm_s": 30.0,
    "water_depth_m": 1.5,
    "cable_tension_n": 25.0,
    "annotator_name": "Amine",
    "notes": "Bonne visibilité"
  },
  "saved_at": "2025-06-01T10:15:00"
}
```

### Format `{stem}.txt` (label YOLO)

```
0 0.266813 0.041759 0.267969 0.082593 0.269010 0.123456 ...
```

Une seule ligne : `classe x1 y1 x2 y2 ... xn yn` (tous les points normalisés entre 0 et 1).

---

## 10. Entraînement IA et modèles

### Entraînement par session

Depuis l'onglet **IA** d'une session :

1. Vérifier qu'au moins **4 images annotées** avec `current_speed_cm_s` renseigné sont présentes
2. Choisir le nombre d'epochs (défaut : 50)
3. Cliquer **« Entraîner »** → l'entraînement tourne en arrière-plan
4. La progression (epoch courante, loss train/val) s'affiche en temps réel
5. À la fin : MAE, RMSE, et graphiques de résultats

Le modèle est sauvegardé dans `data/models/{session_name}_vc_model.pth`.

### Entraînement global

Depuis l'onglet **Global** :
- Sélectionner une ou plusieurs sessions (ou toutes)
- L'entraînement combine toutes les images annotées
- Le modèle est sauvegardé sous `data/models/{model_name}.pth`

### Prédiction sur une nouvelle image

```bash
# Via l'interface : onglet "Prédire" → upload d'image → résultat en cm/s

# Via l'API directement :
curl -X POST http://localhost:8000/api/predict \
  -F "model_name=global_vc_model" \
  -F "file=@mon_image.jpg"
# Réponse : {"vitesse_estimee": 12.34, "model_used": "global_vc_model"}
```

---

## 11. Auto-annotation YOLO

Si un fichier `backend/best.pt` est présent (modèle YOLO entraîné sur des câbles), l'application peut proposer une annotation automatique :

1. Ouvrir une image dans l'éditeur
2. Cliquer **« Prédire »** (ou `?`)
3. Le backend envoie l'image au modèle YOLO
4. Le masque de segmentation détecté est squelettisé → centerline extraite
5. Les points s'affichent sur l'image et peuvent être corrigés manuellement

Vérifier la disponibilité du modèle :
```bash
curl http://localhost:8000/api/yolo/status
# {"model_path": "...", "model_exists": true, "model_loaded": true}
```

---

## 12. Export des données

### Export par session

```bash
GET /api/sessions/{name}/export/download
```

Génère un ZIP avec la structure YOLO standard + CSV des annotations.

### Export global

```bash
GET /api/export/global/download
```

Combine toutes les sessions. Les noms de fichiers sont préfixés par `{session_name}__` pour éviter les collisions.

### Utiliser le dataset exporté avec YOLO

```bash
# Après décompression du ZIP :
yolo segment train data=dataset.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
```

Le fichier `dataset.yaml` est déjà configuré avec les bons chemins et `nc: 1` (1 classe : `cable`).

---

## 13. Étendre / modifier l'application

### Ajouter un champ de condition

1. Dans `backend/main.py`, ajouter le champ dans la liste `fieldnames` de la fonction `export_download`
2. Dans le frontend (`frontend/src/`), ajouter le champ dans le formulaire de l'éditeur d'annotation
3. Le champ sera automatiquement sauvegardé dans `conditions` du JSON et exporté dans le CSV

### Changer le modèle IA

Le backbone peut être remplacé dans la fonction `run_train_session` :

```python
# Remplacer ResNet-18 par ResNet-50 :
model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
```

### Modifier les ports

- Backend : changer `--port 8000` dans `start.sh` et mettre à jour l'URL dans le frontend
- Frontend : modifier `vite.config.ts`

---

## 14. FAQ et dépannage

**L'application ne démarre pas — erreur `uvicorn`**  
→ Vérifier que Python 3.10+ est installé : `python --version`  
→ Réinstaller les dépendances : `pip install -r backend/requirements.txt`

**Le frontend ne s'ouvre pas**  
→ Vérifier Node.js : `node --version` (doit être ≥ 18)  
→ Relancer `cd frontend && npm install`

**L'extraction vidéo ne fonctionne pas**  
→ Vérifier FFmpeg : `ffmpeg -version`  
→ Installer via `brew install ffmpeg` (macOS) ou `apt install ffmpeg` (Linux)

**L'entraînement IA échoue — "PyTorch non installé"**  
→ `pip install torch torchvision`

**L'auto-annotation YOLO retourne "modèle non disponible"**  
→ Placer un fichier `best.pt` dans `backend/best.pt`  
→ Vérifier : `curl http://localhost:8000/api/yolo/status`

**Les annotations existantes changent de nombre de points**  
→ C'est normal. Le rééchantillonnage rétroactif garantit la cohérence du dataset. Toutes les annotations d'une session ont toujours le même nombre de points.

**Où sont stockées les données ?**  
→ Dans `data/sessions/` à la racine du projet. Ce dossier n'est pas versionné (`.gitignore`). Penser à le sauvegarder séparément.

---

## Licence

Développé au **Laboratoire COSMER — Université de Toulon**.  
Usage interne recherche. Contacter le laboratoire pour toute question.
