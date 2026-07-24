# Arabic Khat Classification Web System

Web-based research application to classify Arabic calligraphy styles (`Naskhi`, `Diwani`, `Diwani Jali`, `Tsuluts`) using CNN with VGG16 transfer learning.

## Technology Stack
- Flask + Flask-SQLAlchemy
- TensorFlow / Keras (VGG16 Transfer Learning)
- MySQL (XAMPP) + PyMySQL
- scikit-learn + matplotlib
- Pillow + OpenCV
- Bootstrap 5 (modern dashboard UI)

## Installation

### Windows (Recommended — one-click)

1. Install [Python 3.10/3.11](https://www.python.org/downloads/) (check **Add Python to PATH**)
2. Install [XAMPP](https://www.apachefriends.org/) and start **MySQL**
3. Copy Teachable Machine files to `static/model/` (`model.json`, `metadata.json`, `weights.bin`)
4. Double-click `installer/install_windows.bat`
5. Double-click `installer/run_windows.bat`
6. Open `http://127.0.0.1:5002` — login: `admin` / `admin123`

Full guide: [`installer/README_INSTALL_WINDOWS.md`](installer/README_INSTALL_WINDOWS.md)

### VS Code (PowerShell terminal)

Folder `.venv` **belum ada** sampai Anda menjalankan installer. Jangan copy `.venv` dari Mac.

**Langkah 1 — Install (sekali saja):**

```powershell
cd E:\xampp\htdocs\khat-classification-web
powershell -ExecutionPolicy Bypass -File installer\install_windows.ps1
```

Atau double-click `installer\install_windows.bat`

**Langkah 2 — Pastikan MySQL XAMPP running**

**Langkah 3 — Jalankan aplikasi:**

```powershell
powershell -ExecutionPolicy Bypass -File installer\run_windows.ps1
```

Atau tanpa aktivasi venv (cara paling aman di PowerShell):

```powershell
.\.venv\Scripts\python.exe app.py
```

**VS Code:** Terminal → Run Task → `Arabic Khat: Install (Windows)` lalu `Arabic Khat: Run Server`

**Catatan PowerShell:** Perintah `.\.venv\Scripts\activate` sering gagal. Gunakan `Activate.ps1`:

```powershell
.\.venv\Scripts\Activate.ps1
```

Atau langsung pakai `.\.venv\Scripts\python.exe` tanpa activate.

### macOS / Linux (manual)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # edit DB credentials
python installer/setup_app.py
python app.py
```

Application URL: `http://127.0.0.1:5002` (configure `APP_PORT` in `.env`)

### Repair (Windows)

If dependencies or folders are missing:

```
installer/repair_windows.bat
```

System check:

```bash
python installer/check_system.py
```

## XAMPP MySQL Setup
Use this `.env` configuration:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=db_khat_classification
DB_USER=dbadmin
DB_PASSWORD=DbAdmin@123!
```

## Database Import
1. Start Apache and MySQL from XAMPP Control Panel.
2. Open phpMyAdmin.
3. Create/import database using `database/db_khat_classification.sql`.
4. Run the app. Tables are also auto-created by SQLAlchemy if missing.

## Dataset Preparation

### Import from local folder
Copy calligraphy images from a local source folder into `dataset/raw/` with normalized class names (`naskhi`, `diwani`, `diwani_jali`, `tsuluts`):

```bash
python services/dataset_import_service.py
```

Optional flags:

```bash
python services/dataset_import_service.py --source "/path/to/your/dataset"
python services/dataset_import_service.py --no-db
```

The import script copies images safely (original files are not deleted), renames duplicates if needed, validates supported formats (`.jpg`, `.jpeg`, `.png`, `.webp`), and registers images in the database so **Dataset Management** shows correct class counts.

### Manual upload workflow
1. Place your source dataset zip (`kaligrafi (1).zip`) into the project and extract it, or run the import command above.
2. Use **Dataset Management > Upload Dataset** to add individual images with class labels.
3. Click **Process & Split Dataset** to generate train/validation/test directories (70/15/15 split).

### Image Optimization (Compression)
Large dataset images can slow down previews and training. Use the built-in optimizer to create safe compressed copies without deleting originals.

**From Dataset Management UI:**
1. Open **Dataset Management**.
2. Click **Optimize Images** and confirm.
3. Wait for progress to complete (status card shows compression summary).

**From command line:**
```bash
# Optimize all dataset folders (raw, train, validation, test, processed, processed_balanced)
python scripts/optimize_images.py

# Optimize a single source folder
python scripts/optimize_images.py --source dataset/raw --quality 85 --max-size 1024
python scripts/optimize_images.py --source dataset/train --model-size 224 --model-only
```

Options:
- `--source` — folder under `dataset/` or `all` (default: `all`)
- `--quality` — JPEG quality for display images (default: 85)
- `--max-size` — max dimension for display images (default: 1024)
- `--model-size` — square size for model-ready images (default: 224)
- `--trim-border` — trim near-white borders for model images (`true`/`false`, default: `false`)
- `--force` — overwrite existing optimized files

Output folders:
- `dataset/optimized/display/` — lightweight preview images (max 1024px, JPEG quality 85)
- `dataset/optimized/model/` — model-ready 224×224 padded JPEG copies (quality 90)

Reports:
- `model/image_optimization_report.json`
- `model/image_optimization_status.json`

Original images in `dataset/raw/` and split folders remain unchanged. Thumbnails and previews prefer optimized display copies when available. Training prefers `dataset/optimized/model/train/` when present.

## Model Training

### Training Modes
| Mode | Purpose | Architecture | Epochs |
|------|---------|--------------|--------|
| **Fast Demo** | Quick workflow testing | MobileNetV2 / EfficientNetB0 | 3–5 |
| **Research Accuracy** | Final thesis/research model | EfficientNetB0 (recommended) or VGG16 | Two-phase 10–20 + 5–15 |

### Memory-Efficient Pipeline
- Uses `tf.keras.utils.image_dataset_from_directory()` with `prefetch(AUTOTUNE)` — no RAM cache by default
- Auto batch size fallback: 32 → 16 → 8 → 4 on memory errors
- Optional: use `dataset/optimized/model/` images for faster loading
- Class weights for imbalanced dataset (Naskhi/Diwani minority classes)
- Safe augmentation only (no horizontal flip, no extreme rotation)
- Two-phase training in Research mode: feature extraction + fine-tuning last layers

### Model Files
| File | Purpose |
|------|---------|
| `model/khat_best.keras` | Best validation checkpoint (used for prediction) |
| `model/khat_latest.keras` | Most recent training run |
| `model/khat_vgg16_model.keras` | Legacy compatibility path |
| `model/model_metadata.json` | Architecture, mode, validation accuracy |
| `model/class_weights.json` | Class weight values |
| `model/training_history.json` | Full training log |

### Steps
1. Open **Manajemen Model** (`/model-management/`).
2. Ensure dataset shows **ready for training** (train/validation/test folders populated).
3. Select **Research Accuracy Mode** + **EfficientNetB0** for best results.
4. Click **Start Training**.
5. Run **Evaluation** on the held-out test set (metrics are real, not faked).

**TensorFlow requirement:**
```bash
pip uninstall tensorflow keras tensorflow-macos -y
pip install tensorflow==2.15.0
```

If training fails, check `logs/training_error.log` for the full traceback.

## Model Evaluation
1. Open **Evaluation**.
2. Click **Run Evaluation**.
3. Metrics and charts are generated and saved under `static/evaluation/`.

## Image Classification
1. Open **Classify Image**.
2. Upload one calligraphy image.
3. The system predicts Khat type and confidence scores for all classes.
4. Results are saved in `classification_results`.

## Default Admin Account
- Username: `admin`
- Password: `admin123`

## Notes
- Passwords are hashed with Werkzeug secure hashing.
- The system validates file types and handles missing model/dataset errors with flash messages.
