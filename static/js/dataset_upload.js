function formatFileSize(bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
    }
    if (bytes >= 1024 * 1024) {
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }
    return (bytes / 1024).toFixed(1) + " KB";
}

function showZipAlert(title, message, tone) {
    const alertEl = document.getElementById("zipUploadAlert");
    if (!alertEl) return;
    alertEl.className = "upload-inline-alert alert-" + (tone || "warning");
    alertEl.innerHTML = title
        ? "<strong>" + title + "</strong>" + message
        : message;
    alertEl.classList.remove("d-none");
}

function hideZipAlert() {
    const alertEl = document.getElementById("zipUploadAlert");
    if (alertEl) alertEl.classList.add("d-none");
}

function setZipProgress(percent, statusText, subStatusText) {
    const wrap = document.getElementById("zipUploadProgressWrap");
    const bar = document.getElementById("zipUploadProgressBar");
    const label = document.getElementById("zipUploadPercent");
    const statusEl = document.getElementById("zipUploadStatusText");
    const subStatusEl = document.getElementById("zipUploadSubStatus");
    if (!wrap || !bar || !label) return;
    const safe = Math.max(0, Math.min(100, percent));
    wrap.classList.remove("d-none");
    bar.style.width = safe + "%";
    label.textContent = safe.toFixed(0) + "%";
    if (statusEl && statusText) statusEl.innerHTML = statusText;
    if (subStatusEl && subStatusText) subStatusEl.textContent = subStatusText;
}

function resetZipProgress() {
    const wrap = document.getElementById("zipUploadProgressWrap");
    const bar = document.getElementById("zipUploadProgressBar");
    const label = document.getElementById("zipUploadPercent");
    const subStatusEl = document.getElementById("zipUploadSubStatus");
    if (wrap) wrap.classList.add("d-none");
    if (bar) bar.style.width = "0%";
    if (label) label.textContent = "0%";
    if (subStatusEl) subStatusEl.textContent = "Mohon tunggu, jangan tutup halaman.";
}

function uploadZipWithProgress(file, onProgress) {
    const chunkBytes = Number(window.DATASET_UPLOAD_CHUNK_BYTES || 20 * 1024 * 1024);
    if (file.size > chunkBytes && window.DATASET_ZIP_CHUNK_URL && window.DATASET_ZIP_FINALIZE_URL) {
        return uploadZipInChunks(file, chunkBytes, onProgress);
    }
    return uploadZipSingleRequest(file, onProgress);
}

function uploadZipSingleRequest(file, onProgress) {
    return new Promise(function (resolve, reject) {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append("dataset_zip", file);

        xhr.open("POST", window.DATASET_ZIP_UPLOAD_URL, true);
        xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
        xhr.timeout = 0;

        xhr.upload.addEventListener("progress", function (event) {
            if (!event.lengthComputable) return;
            const percent = Math.min((event.loaded / event.total) * 85, 85);
            onProgress(
                percent,
                '<i class="bi bi-arrow-up-circle" aria-hidden="true"></i> Uploading...',
                file.name + " · " + formatFileSize(file.size)
            );
        });

        xhr.upload.addEventListener("load", function () {
            onProgress(
                90,
                '<i class="bi bi-gear-wide-connected" aria-hidden="true"></i> Extracting &amp; optimizing...',
                "Memproses gambar satu per satu. Mohon tunggu."
            );
        });

        xhr.addEventListener("load", function () {
            let payload = null;
            try {
                payload = JSON.parse(xhr.responseText || "{}");
            } catch (error) {
                reject(new Error("Respons server tidak valid."));
                return;
            }
            if (xhr.status >= 200 && xhr.status < 300) {
                onProgress(
                    100,
                    '<i class="bi bi-check-circle" aria-hidden="true"></i> Completed',
                    "Import selesai. Mengalihkan ke halaman dataset..."
                );
                resolve(payload);
                return;
            }
            reject(new Error(payload.message || "Upload ZIP gagal."));
        });

        xhr.addEventListener("error", function () {
            reject(new Error("Koneksi upload terputus. Silakan coba lagi."));
        });

        xhr.send(formData);
    });
}

function uploadChunkBlob(url, formData) {
    return new Promise(function (resolve, reject) {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", url, true);
        xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
        xhr.timeout = 0;
        xhr.addEventListener("load", function () {
            let payload = null;
            try {
                payload = JSON.parse(xhr.responseText || "{}");
            } catch (error) {
                reject(new Error("Respons server tidak valid."));
                return;
            }
            if (xhr.status >= 200 && xhr.status < 300 && payload.success !== false) {
                resolve(payload);
                return;
            }
            reject(new Error((payload && payload.message) || "Upload chunk gagal."));
        });
        xhr.addEventListener("error", function () {
            reject(new Error("Koneksi upload terputus. Silakan coba lagi."));
        });
        xhr.send(formData);
    });
}

function uploadZipInChunks(file, chunkBytes, onProgress) {
    const totalChunks = Math.ceil(file.size / chunkBytes);
    let sessionId = "";
    let uploadedBytes = 0;

    function sendChunk(index) {
        if (index >= totalChunks) {
            onProgress(
                90,
                '<i class="bi bi-gear-wide-connected" aria-hidden="true"></i> Extracting &amp; optimizing...',
                "Semua chunk diterima. Memproses dataset..."
            );
            const finalizeData = new FormData();
            finalizeData.append("session_id", sessionId);
            return uploadChunkBlob(window.DATASET_ZIP_FINALIZE_URL, finalizeData).then(function (payload) {
                onProgress(
                    100,
                    '<i class="bi bi-check-circle" aria-hidden="true"></i> Completed',
                    "Import selesai. Mengalihkan ke halaman dataset..."
                );
                return payload;
            });
        }

        const start = index * chunkBytes;
        const end = Math.min(start + chunkBytes, file.size);
        const blob = file.slice(start, end);
        const formData = new FormData();
        formData.append("chunk", blob, file.name + ".part" + index);
        formData.append("chunk_index", String(index));
        formData.append("total_chunks", String(totalChunks));
        formData.append("filename", file.name);
        if (sessionId) formData.append("session_id", sessionId);

        return uploadChunkBlob(window.DATASET_ZIP_CHUNK_URL, formData).then(function (payload) {
            if (payload.session_id) sessionId = payload.session_id;
            uploadedBytes = end;
            const percent = Math.min((uploadedBytes / file.size) * 85, 85);
            onProgress(
                percent,
                '<i class="bi bi-arrow-up-circle" aria-hidden="true"></i> Uploading chunks...',
                "Chunk " + (index + 1) + " / " + totalChunks + " · " + formatFileSize(uploadedBytes)
            );
            return sendChunk(index + 1);
        });
    }

    return sendChunk(0);
}

function bindDropzone(dropzone, input, onFiles) {
    if (!dropzone || !input) return;

    const openPicker = function () {
        input.click();
    };

    dropzone.addEventListener("click", function (e) {
        if (e.target === input) return;
        openPicker();
    });

    dropzone.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openPicker();
        }
    });

    input.addEventListener("change", function () {
        onFiles(input.files);
    });

    ["dragenter", "dragover"].forEach(function (ev) {
        dropzone.addEventListener(ev, function (e) {
            e.preventDefault();
            dropzone.classList.add("drag-active");
        });
    });

    ["dragleave", "drop"].forEach(function (ev) {
        dropzone.addEventListener(ev, function (e) {
            e.preventDefault();
            dropzone.classList.remove("drag-active");
        });
    });

    dropzone.addEventListener("drop", function (e) {
        const files = e.dataTransfer.files;
        if (!files.length) return;
        const dt = new DataTransfer();
        Array.from(files).forEach(function (f) {
            dt.items.add(f);
        });
        input.files = dt.files;
        onFiles(input.files);
    });
}

function getFormatSummary(files) {
    const formats = {};
    Array.from(files).forEach(function (file) {
        const ext = (file.name.split(".").pop() || "").toUpperCase();
        formats[ext] = (formats[ext] || 0) + 1;
    });
    return Object.keys(formats)
        .map(function (ext) {
            return ext + " (" + formats[ext] + ")";
        })
        .join(", ");
}

document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("datasetImageInput");
    const dropzone = document.getElementById("uploadDropzone");
    const dropzoneInner = document.getElementById("uploadDropzoneInner");
    const imageSelectedSummary = document.getElementById("imageSelectedSummary");
    const imageSelectedCount = document.getElementById("imageSelectedCount");
    const imageSelectedSize = document.getElementById("imageSelectedSize");
    const previewEmpty = document.getElementById("previewEmpty");
    const previewContent = document.getElementById("previewContent");
    const previewImage = document.getElementById("previewImage");
    const previewFileCount = document.getElementById("previewFileCount");
    const previewTotalSize = document.getElementById("previewTotalSize");
    const previewClassLabel = document.getElementById("previewClassLabel");
    const previewFormatSummary = document.getElementById("previewFormatSummary");
    const filePreviewList = document.getElementById("filePreviewList");
    const classSelect = document.getElementById("classNameSelect");
    const uploadForm = document.getElementById("uploadDatasetForm");
    const uploadBtn = document.getElementById("uploadSubmitBtn");

    const zipInput = document.getElementById("datasetZipInput");
    const zipDropzone = document.getElementById("zipDropzone");
    const zipDropzoneInner = document.getElementById("zipDropzoneInner");
    const zipSelectedSummary = document.getElementById("zipSelectedSummary");
    const zipSelectedName = document.getElementById("zipSelectedName");
    const zipSelectedSize = document.getElementById("zipSelectedSize");
    const zipForm = document.getElementById("zipUploadForm");
    const zipBtn = document.getElementById("zipUploadBtn");
    const maxBytes = Number(window.DATASET_UPLOAD_MAX_BYTES || 1073741824);

    function updateImagePreview(files) {
        const hasFiles = files && files.length > 0;
        if (uploadBtn) uploadBtn.disabled = !hasFiles;

        if (!hasFiles) {
            if (previewEmpty) previewEmpty.classList.remove("d-none");
            if (previewContent) previewContent.classList.add("d-none");
            if (dropzone) dropzone.classList.remove("has-file");
            if (dropzoneInner) dropzoneInner.classList.remove("d-none");
            if (imageSelectedSummary) imageSelectedSummary.classList.add("d-none");
            return;
        }

        let totalSize = 0;
        Array.from(files).forEach(function (f) {
            totalSize += f.size;
        });

        if (dropzone) dropzone.classList.add("has-file");
        if (dropzoneInner) dropzoneInner.classList.add("d-none");
        if (imageSelectedSummary) imageSelectedSummary.classList.remove("d-none");
        if (imageSelectedCount) {
            imageSelectedCount.textContent =
                files.length + " file" + (files.length > 1 ? "" : "") + " dipilih";
        }
        if (imageSelectedSize) imageSelectedSize.textContent = formatFileSize(totalSize);

        if (previewEmpty) previewEmpty.classList.add("d-none");
        if (previewContent) previewContent.classList.remove("d-none");
        if (previewFileCount) previewFileCount.textContent = String(files.length);
        if (previewTotalSize) previewTotalSize.textContent = formatFileSize(totalSize);
        if (previewClassLabel && classSelect) {
            previewClassLabel.textContent =
                classSelect.options[classSelect.selectedIndex].textContent;
        }
        if (previewFormatSummary) previewFormatSummary.textContent = getFormatSummary(files);

        if (filePreviewList) {
            filePreviewList.innerHTML = "";
            Array.from(files).forEach(function (file, index) {
                const item = document.createElement("div");
                item.className = "upload-thumb-item";
                if (file.type.startsWith("image/")) {
                    const img = document.createElement("img");
                    img.alt = file.name;
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        img.src = e.target.result;
                    };
                    reader.readAsDataURL(file);
                    item.appendChild(img);
                    if (index === 0 && previewImage) {
                        const mainReader = new FileReader();
                        mainReader.onload = function (e) {
                            previewImage.src = e.target.result;
                        };
                        mainReader.readAsDataURL(file);
                    }
                } else {
                    item.classList.add("file-only");
                    item.innerHTML =
                        '<i class="bi bi-file-image" aria-hidden="true"></i>' + file.name;
                }
                filePreviewList.appendChild(item);
            });
        }
    }

    if (classSelect) {
        classSelect.addEventListener("change", function () {
            if (input && input.files && input.files.length && previewClassLabel) {
                previewClassLabel.textContent =
                    classSelect.options[classSelect.selectedIndex].textContent;
            }
        });
    }

    bindDropzone(dropzone, input, updateImagePreview);

    if (uploadForm && uploadBtn) {
        uploadForm.addEventListener("submit", function () {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML =
                '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span><span>Uploading...</span>';
        });
    }

    function updateZipSelection(file) {
        if (!file) {
            if (zipDropzone) zipDropzone.classList.remove("has-file");
            if (zipDropzoneInner) zipDropzoneInner.classList.remove("d-none");
            if (zipSelectedSummary) zipSelectedSummary.classList.add("d-none");
            return;
        }
        if (zipDropzone) zipDropzone.classList.add("has-file");
        if (zipDropzoneInner) zipDropzoneInner.classList.add("d-none");
        if (zipSelectedSummary) zipSelectedSummary.classList.remove("d-none");
        if (zipSelectedName) zipSelectedName.textContent = file.name;
        if (zipSelectedSize) zipSelectedSize.textContent = formatFileSize(file.size);

        if (file.size > maxBytes) {
            showZipAlert(
                "File terlalu besar",
                "Ukuran file melebihi batas " +
                    (window.DATASET_UPLOAD_MAX_MB || 1024) +
                    " MB. Silakan kompres gambar atau bagi dataset menjadi beberapa ZIP.",
                "danger"
            );
        } else {
            hideZipAlert();
        }
    }

    bindDropzone(zipDropzone, zipInput, function (files) {
        updateZipSelection(files && files[0] ? files[0] : null);
    });

    if (zipForm && zipBtn && zipInput) {
        zipForm.addEventListener("submit", function (event) {
            event.preventDefault();
            hideZipAlert();

            const file = zipInput.files && zipInput.files[0];
            if (!file) {
                showZipAlert(
                    "File belum dipilih",
                    "Silakan pilih file ZIP dataset terlebih dahulu.",
                    "warning"
                );
                return;
            }
            if (!file.name.toLowerCase().endsWith(".zip")) {
                showZipAlert(
                    "Format tidak valid",
                    "Format file harus .zip. Pastikan Anda mengunggah arsip dataset yang benar.",
                    "warning"
                );
                return;
            }
            if (file.size > maxBytes) {
                showZipAlert(
                    "File terlalu besar",
                    "Ukuran file melebihi batas " +
                        (window.DATASET_UPLOAD_MAX_MB || 1024) +
                        " MB. Silakan kompres gambar atau bagi dataset menjadi beberapa ZIP.",
                    "danger"
                );
                return;
            }

            zipBtn.disabled = true;
            zipBtn.innerHTML =
                '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span><span>Processing...</span>';
            setZipProgress(
                0,
                '<i class="bi bi-arrow-up-circle" aria-hidden="true"></i> Uploading...',
                file.name + " · " + formatFileSize(file.size)
            );

            uploadZipWithProgress(file, setZipProgress)
                .then(function (payload) {
                    if (payload.redirect_url) {
                        window.location.href = payload.redirect_url;
                        return;
                    }
                    showZipAlert("Import selesai", "Dataset ZIP berhasil diproses.", "success");
                })
                .catch(function (error) {
                    showZipAlert("Upload gagal", error.message || "Upload ZIP gagal.", "danger");
                })
                .finally(function () {
                    zipBtn.disabled = false;
                    zipBtn.innerHTML =
                        '<i class="bi bi-file-earmark-arrow-up" aria-hidden="true"></i><span>Import ZIP Dataset</span>';
                    resetZipProgress();
                });
        });
    }
});
