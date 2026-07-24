-- Arabic Khat Classification — skema relasional 9 tabel (MySQL/MariaDB)
CREATE DATABASE IF NOT EXISTS db_khat_classification
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE db_khat_classification;

-- 1. pengguna
CREATE TABLE IF NOT EXISTS pengguna (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nama_lengkap VARCHAR(100) NOT NULL,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(120) NULL,
    password_hash VARCHAR(255) NOT NULL,
    peran VARCHAR(20) NOT NULL DEFAULT 'pengguna',
    status VARCHAR(20) NOT NULL DEFAULT 'aktif',
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    diperbarui_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. kelas_khat
CREATE TABLE IF NOT EXISTS kelas_khat (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nama_kelas VARCHAR(100) NOT NULL,
    slug VARCHAR(50) NOT NULL UNIQUE,
    deskripsi TEXT NULL,
    ciri_utama TEXT NULL,
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    diperbarui_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. gambar_dataset
CREATE TABLE IF NOT EXISTS gambar_dataset (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_kelas INT NOT NULL,
    nama_file VARCHAR(255) NOT NULL,
    nama_file_asli VARCHAR(255) NOT NULL,
    path_asli VARCHAR(512) NOT NULL,
    path_proses VARCHAR(512) NULL,
    path_model_ready VARCHAR(512) NULL,
    format_file VARCHAR(20) NULL,
    ukuran_file INT NULL,
    lebar INT NULL,
    tinggi INT NULL,
    sumber_data VARCHAR(50) NULL DEFAULT 'raw',
    hash_file VARCHAR(64) NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'active',
    split_data VARCHAR(20) NULL,
    id_legacy INT NULL UNIQUE,
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    diperbarui_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_gambar_kelas FOREIGN KEY (id_kelas) REFERENCES kelas_khat(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. versi_model
CREATE TABLE IF NOT EXISTS versi_model (
    id INT AUTO_INCREMENT PRIMARY KEY,
    versi VARCHAR(50) NOT NULL,
    nama_model VARCHAR(255) NOT NULL,
    sumber_model VARCHAR(100) NOT NULL,
    runtime VARCHAR(100) NULL,
    file_model VARCHAR(512) NULL,
    file_metadata VARCHAR(512) NULL,
    file_weights VARCHAR(512) NULL,
    lebar_input INT NULL DEFAULT 224,
    tinggi_input INT NULL DEFAULT 224,
    label_json TEXT NULL,
    status VARCHAR(30) NULL DEFAULT 'active',
    akurasi FLOAT NULL,
    presisi FLOAT NULL,
    recall FLOAT NULL,
    f1_score FLOAT NULL,
    jumlah_kelas INT NULL DEFAULT 4,
    jumlah_gambar_dataset INT NULL,
    catatan TEXT NULL,
    aktif TINYINT(1) NOT NULL DEFAULT 0,
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    diaktifkan_pada DATETIME NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. riwayat_klasifikasi
CREATE TABLE IF NOT EXISTS riwayat_klasifikasi (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_pengguna INT NULL,
    id_model INT NULL,
    id_gambar_dataset INT NULL,
    nama_file_upload VARCHAR(255) NULL,
    path_upload VARCHAR(512) NULL,
    id_kelas_diharapkan INT NULL,
    id_kelas_prediksi INT NULL,
    sumber_kelas_diharapkan VARCHAR(50) NULL,
    confidence FLOAT NULL,
    id_kelas_top2 INT NULL,
    skor_top2 FLOAT NULL,
    margin_top2 FLOAT NULL,
    status_validasi VARCHAR(100) NULL,
    keputusan_akhir VARCHAR(100) NULL,
    sumber_input VARCHAR(50) NULL DEFAULT 'upload',
    id_legacy INT NULL UNIQUE,
    data_ekstra_json TEXT NULL,
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_riwayat_pengguna FOREIGN KEY (id_pengguna) REFERENCES pengguna(id),
    CONSTRAINT fk_riwayat_model FOREIGN KEY (id_model) REFERENCES versi_model(id),
    CONSTRAINT fk_riwayat_gambar FOREIGN KEY (id_gambar_dataset) REFERENCES gambar_dataset(id),
    CONSTRAINT fk_riwayat_kelas_diharapkan FOREIGN KEY (id_kelas_diharapkan) REFERENCES kelas_khat(id),
    CONSTRAINT fk_riwayat_kelas_prediksi FOREIGN KEY (id_kelas_prediksi) REFERENCES kelas_khat(id),
    CONSTRAINT fk_riwayat_kelas_top2 FOREIGN KEY (id_kelas_top2) REFERENCES kelas_khat(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. probabilitas_prediksi
CREATE TABLE IF NOT EXISTS probabilitas_prediksi (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_riwayat INT NOT NULL,
    id_kelas INT NOT NULL,
    skor_softmax FLOAT NOT NULL,
    persentase_probabilitas FLOAT NOT NULL,
    urutan_ranking INT NOT NULL,
    CONSTRAINT fk_prob_riwayat FOREIGN KEY (id_riwayat) REFERENCES riwayat_klasifikasi(id) ON DELETE CASCADE,
    CONSTRAINT fk_prob_kelas FOREIGN KEY (id_kelas) REFERENCES kelas_khat(id),
    CONSTRAINT uq_prob_pred_kelas UNIQUE (id_riwayat, id_kelas)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. perhitungan_algoritma
CREATE TABLE IF NOT EXISTS perhitungan_algoritma (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_riwayat INT NOT NULL UNIQUE,
    lebar_asli INT NULL,
    tinggi_asli INT NULL,
    lebar_proses INT NULL,
    tinggi_proses INT NULL,
    ukuran_input_model VARCHAR(20) NULL,
    bentuk_tensor VARCHAR(50) NULL,
    langkah_preprocessing_json TEXT NULL,
    probabilitas_khat FLOAT NULL,
    probabilitas_non_khat FLOAT NULL,
    keputusan_tahap1 VARCHAR(100) NULL,
    level_confidence VARCHAR(100) NULL,
    status_kelas_dikenal VARCHAR(100) NULL,
    rumus_json TEXT NULL,
    catatan_perhitungan TEXT NULL,
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_perhitungan_riwayat FOREIGN KEY (id_riwayat) REFERENCES riwayat_klasifikasi(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. evaluasi_model
CREATE TABLE IF NOT EXISTS evaluasi_model (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_model INT NULL,
    id_induk INT NULL,
    id_gambar_dataset INT NULL,
    id_kelas_asli INT NULL,
    id_kelas_prediksi INT NULL,
    confidence FLOAT NULL,
    id_kelas_top2 INT NULL,
    skor_top2 FLOAT NULL,
    margin_top2 FLOAT NULL,
    benar TINYINT(1) NULL,
    jenis_error VARCHAR(100) NULL,
    rekomendasi TEXT NULL,
    url_gambar VARCHAR(512) NULL,
    nama_file_simpan VARCHAR(255) NULL,
    total_gambar INT NULL,
    prediksi_benar INT NULL,
    prediksi_salah INT NULL,
    akurasi FLOAT NULL,
    presisi FLOAT NULL,
    recall FLOAT NULL,
    f1_score FLOAT NULL,
    tipe_record VARCHAR(30) NULL,
    data_ekstra_json TEXT NULL,
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_eval_model FOREIGN KEY (id_model) REFERENCES versi_model(id),
    CONSTRAINT fk_eval_induk FOREIGN KEY (id_induk) REFERENCES evaluasi_model(id),
    CONSTRAINT fk_eval_gambar FOREIGN KEY (id_gambar_dataset) REFERENCES gambar_dataset(id),
    CONSTRAINT fk_eval_kelas_asli FOREIGN KEY (id_kelas_asli) REFERENCES kelas_khat(id),
    CONSTRAINT fk_eval_kelas_prediksi FOREIGN KEY (id_kelas_prediksi) REFERENCES kelas_khat(id),
    CONSTRAINT fk_eval_kelas_top2 FOREIGN KEY (id_kelas_top2) REFERENCES kelas_khat(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. log_koreksi
CREATE TABLE IF NOT EXISTS log_koreksi (
    id INT AUTO_INCREMENT PRIMARY KEY,
    id_riwayat INT NOT NULL,
    id_pengguna INT NULL,
    id_kelas_sebelumnya INT NULL,
    id_kelas_benar INT NULL,
    catatan TEXT NULL,
    dibuat_pada DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_koreksi_riwayat FOREIGN KEY (id_riwayat) REFERENCES riwayat_klasifikasi(id),
    CONSTRAINT fk_koreksi_pengguna FOREIGN KEY (id_pengguna) REFERENCES pengguna(id),
    CONSTRAINT fk_koreksi_sebelum FOREIGN KEY (id_kelas_sebelumnya) REFERENCES kelas_khat(id),
    CONSTRAINT fk_koreksi_benar FOREIGN KEY (id_kelas_benar) REFERENCES kelas_khat(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Master data: empat kelas khat
INSERT INTO kelas_khat (nama_kelas, slug, deskripsi) VALUES
    ('Khat Naskhi', 'naskhi', 'Gaya tulisan Arab yang paling umum dan mudah dibaca.'),
    ('Khat Riq''ah', 'riqah', 'Gaya tulisan Arab yang ringkas untuk keperluan sehari-hari.'),
    ('Khat Diwani', 'diwani', 'Gaya kaligrafi Arab yang dekoratif dan mengalir.'),
    ('Khat Kufi', 'kufi', 'Gaya kaligrafi Arab geometris bersejarah.')
ON DUPLICATE KEY UPDATE nama_kelas = VALUES(nama_kelas);

-- Pengguna demo (password: user123 — ganti setelah instalasi)
INSERT INTO pengguna (nama_lengkap, username, password_hash, peran, status)
SELECT 'Pengguna Demo', 'user',
       'pbkdf2:sha256:1000000$x4n8XKRm3dxN6FdR$c0760372f3d2f8e199962e42a89d00b2b512da952a841c6e55acfa45ec682b58',
       'pengguna', 'aktif'
WHERE NOT EXISTS (SELECT 1 FROM pengguna WHERE username = 'user');
