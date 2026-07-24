# Panduan Instalasi Arabic Khat AI di Windows

Aplikasi **Arabic Khat AI** adalah sistem klasifikasi kaligrafi Arab (Naskhi, Diwani, Diwani Jali, Tsuluts) berbasis Flask.

Dokumen ini menjelaskan cara install, menjalankan, dan memperbaiki aplikasi di Windows **tanpa konfigurasi manual**.

---

## 1. Persyaratan Sistem

| Item | Minimum |
|------|---------|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10 atau 3.11 (centang **Add Python to PATH**) |
| RAM | 8 GB (16 GB direkomendasikan jika training model) |
| Disk | 5 GB ruang kosong |
| Database | MySQL via **XAMPP** (MariaDB) |
| Browser | Chrome / Edge / Firefox |

**Catatan:** Aplikasi ini menggunakan **MySQL**, bukan SQLite. Install **XAMPP for Windows** dan jalankan service **MySQL** sebelum menjalankan aplikasi.

---

## 2. Cara Install Pertama Kali

### Step 1 — Install Python

1. Download dari [python.org/downloads](https://www.python.org/downloads/)
2. Pilih Python **3.10** atau **3.11**
3. Centang **Add Python to PATH**
4. Klik Install

### Step 2 — Install XAMPP (MySQL)

1. Download [XAMPP for Windows](https://www.apachefriends.org/)
2. Install dan buka **XAMPP Control Panel**
3. Start **MySQL**
4. (Opsional) Import database dari `database/db_khat_classification.sql` via phpMyAdmin

### Step 3 — Extract Project

Extract folder proyek ke lokasi yang aman, misalnya:

```
C:\Users\NamaAnda\Documents\khat-classification-web
```

**Jangan** letakkan di `C:\Program Files\` (bisa permission denied).

### Step 4 — Salin Model Teachable Machine

Salin file model ke:

```
static/model/model.json
static/model/metadata.json
static/model/weights.bin
```

### Step 5 — Jalankan Installer

Double-click:

```
installer/install_windows.bat
```

Tunggu hingga muncul pesan **Instalasi selesai**.

### Step 6 — Jalankan Aplikasi

Double-click:

```
installer/run_windows.bat
```

Browser akan terbuka otomatis di:

```
http://127.0.0.1:5002
```

### Login Default

| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `admin123` |

**PENTING:** Segera ganti password admin setelah login pertama.

---

## 3. Cara Menjalankan Aplikasi (Setelah Install)

1. Pastikan **MySQL XAMPP** sudah running
2. Double-click `installer/run_windows.bat`
3. Buka browser: `http://127.0.0.1:5002`

### VS Code / PowerShell

Jika terminal VS Code menampilkan `ModuleNotFoundError: No module named 'flask'`, artinya **`.venv` belum dibuat** atau belum terinstall dependencies.

**Install (sekali):**

```powershell
cd E:\xampp\htdocs\khat-classification-web
powershell -ExecutionPolicy Bypass -File installer\install_windows.ps1
```

**Jalankan:**

```powershell
powershell -ExecutionPolicy Bypass -File installer\run_windows.ps1
```

Atau:

```powershell
.\.venv\Scripts\python.exe app.py
```

**Jangan** gunakan `python app.py` langsung tanpa `.venv` — Python global tidak punya Flask.

Perintah `.\.venv\Scripts\activate` di PowerShell sering gagal. Gunakan `Activate.ps1` atau jalankan `python.exe` di dalam `.venv\Scripts\` langsung.

### Shortcut Desktop (Opsional)

Jalankan PowerShell di folder proyek:

```powershell
powershell -ExecutionPolicy Bypass -File installer\create_shortcut.ps1
```

Shortcut **Arabic Khat AI** akan muncul di Desktop.

---

## 4. Cara Memperbaiki Jika Error

Double-click:

```
installer/repair_windows.bat
```

Script ini akan:
- Reinstall dependencies Python
- Membuat ulang folder yang hilang
- Memvalidasi file model
- Membersihkan file temporary (`temp/`, `dataset/tmp/`)
- **Tidak menghapus** dataset atau model

Laporan disimpan di `logs/repair.log` dan `logs/install.log`.

---

## 5. Struktur Folder Penting

```
khat-classification-web/
├── app.py                  # Entry point Flask
├── .env                    # Konfigurasi (dibuat otomatis)
├── .venv/                  # Virtual environment Python
├── installer/              # Script install & run Windows
├── static/
│   ├── model/              # Model Teachable Machine (wajib)
│   └── uploads/            # Gambar upload klasifikasi
├── dataset/
│   ├── raw/                # Dataset mentah per kelas
│   ├── processed/          # Dataset olahan
│   └── model_ready/        # Dataset siap training
├── model/                  # Model Keras & laporan
├── logs/                   # Log instalasi & repair
├── temp/                   # File sementara
└── templates/              # HTML Flask
```

---

## 6. Cara Mengganti Model Teachable Machine

1. Export model dari [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Salin 3 file ke `static/model/`:
   - `model.json`
   - `metadata.json`
   - `weights.bin`
3. Restart aplikasi (`run_windows.bat`)

---

## 7. Cara Memindahkan ke Laptop Lain

1. Copy **seluruh folder** proyek (termasuk `dataset/` dan `static/model/` jika ada)
2. Di laptop baru: install Python + XAMPP
3. Jalankan `installer/install_windows.bat`
4. Sesuaikan `.env` jika perlu (password MySQL)
5. Jalankan `installer/run_windows.bat`

**Tips:** Jangan copy folder `.venv` antar mesin — biarkan installer membuat ulang.

---

## 8. Error Umum dan Solusi

### `python is not recognized`

**Penyebab:** Python belum di PATH.

**Solusi:** Install ulang Python dengan centang **Add Python to PATH**, atau gunakan `py -3` dari Python Launcher.

---

### `ModuleNotFoundError: No module named 'flask'`

**Penyebab:** Dependencies belum terinstall atau venv tidak aktif.

**Solusi:** Jalankan `installer/repair_windows.bat` atau `installer/install_windows.bat`.

---

### `model.json not found` / Model TM tidak aktif

**Penyebab:** File model belum disalin.

**Solusi:** Salin `model.json`, `metadata.json`, `weights.bin` ke `static/model/`.

---

### `MySQL connection failed`

**Penyebab:** MySQL XAMPP belum running atau kredensial `.env` salah.

**Solusi:**
1. Start MySQL di XAMPP Control Panel
2. Edit `.env`:
   ```env
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=
   DB_NAME=db_khat_classification
   ```
3. Jalankan `installer/repair_windows.bat`

---

### Port already in use (5002)

**Penyebab:** Port sudah dipakai aplikasi lain.

**Solusi:** Edit `.env`:
```env
APP_PORT=5003
```
Lalu restart `run_windows.bat`.

---

### Permission denied

**Penyebab:** Folder proyek di lokasi terproteksi.

**Solusi:** Pindahkan ke `Documents` atau `Desktop`, bukan `Program Files`.

---

## 9. Konfigurasi `.env`

File `.env` dibuat otomatis dari `.env.example` saat install.

```env
FLASK_APP=app.py
FLASK_ENV=development
APP_HOST=127.0.0.1
APP_PORT=5002
SECRET_KEY=change-this-secret-key

DB_HOST=localhost
DB_PORT=3306
DB_NAME=db_khat_classification
DB_USER=root
DB_PASSWORD=

MAX_UPLOAD_MB=1024
```

---

## 10. Verifikasi Manual

Dari folder proyek (setelah aktivasi `.venv`):

```bat
python installer\check_system.py
```

Semua item `[OK]` = siap digunakan.

---

## Kontak & Dukungan

Jika masih error setelah repair:
1. Baca output di terminal `run_windows.bat`
2. Periksa `logs/install.log` dan `logs/repair.log`
3. Jalankan `python installer\check_system.py` dan catat item `[ERROR]`
