window.datasetThumbError = function (img) {
    img.onerror = null;
    img.classList.add("is-missing");
    img.src = img.dataset.placeholder || window.DATASET_PLACEHOLDER || "";
};

document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
        new bootstrap.Tooltip(el);
    });

    const viewModalEl = document.getElementById("viewImageModal");
    const deleteModalEl = document.getElementById("deleteConfirmModal");
    const deleteForm = document.getElementById("deleteImageForm");
    const modalDeleteBtn = document.getElementById("modalDeleteBtn");

    const viewImageSrc = document.getElementById("viewImageSrc");
    const viewImageEmpty = document.getElementById("viewImageEmpty");
    const viewImageLoading = document.getElementById("viewImageLoading");

    let viewModal;
    let deleteModal;

    function setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value || "-";
    }

    function openViewModal(btn) {
        if (!viewModalEl) return;
        if (!viewModal) viewModal = new bootstrap.Modal(viewModalEl);

        const exists = btn.dataset.exists === "1";
        const imageUrl = btn.dataset.imageUrl;
        const cls = btn.dataset.class || "Unknown";
        const filename = btn.dataset.filename || "Unknown";
        const previewSource = btn.dataset.previewSource || "-";

        setText("viewImageTitle", filename);
        const badge = document.getElementById("viewImageClassBadge");
        if (badge) {
            badge.textContent = cls;
            badge.className = "class-pill " + (btn.dataset.classKey || cls.toLowerCase().replace(/ /g, "_"));
        }

        setText("metaClass", cls);
        setText("metaFilename", filename);
        setText("metaFormat", btn.dataset.format);
        setText("metaSize", btn.dataset.size);
        setText("metaFileSize", btn.dataset.filesize);
        setText("metaSource", btn.dataset.source);
        setText("metaDate", btn.dataset.date);
        setText("metaId", btn.dataset.id);
        setText("metaStatus", exists ? "Available" : "Missing on disk");
        setText("metaPreviewSource", previewSource);
        setText("metaPath", btn.dataset.path || "-");
        setText("metaResolved", btn.dataset.resolved || "-");

        viewImageSrc.classList.add("d-none");
        viewImageEmpty.classList.add("d-none");
        viewImageLoading.classList.remove("d-none");
        viewImageSrc.src = "";

        if (modalDeleteBtn && deleteForm) {
            modalDeleteBtn.classList.remove("d-none");
            modalDeleteBtn.onclick = function () {
                viewModal.hide();
                openDeleteModal(btn);
            };
        }

        viewModal.show();

        if (!exists) {
            viewImageLoading.classList.add("d-none");
            viewImageEmpty.classList.remove("d-none");
            return;
        }

        viewImageSrc.onload = function () {
            viewImageLoading.classList.add("d-none");
            viewImageSrc.classList.remove("d-none");
            viewImageEmpty.classList.add("d-none");
        };
        viewImageSrc.onerror = function () {
            viewImageLoading.classList.add("d-none");
            viewImageSrc.classList.add("d-none");
            viewImageEmpty.classList.remove("d-none");
        };
        viewImageSrc.src = imageUrl + (imageUrl.indexOf("?") >= 0 ? "&" : "?") + "t=" + Date.now();
    }

    function openDeleteModal(btn) {
        if (!deleteModalEl || !deleteForm) return;
        if (!deleteModal) deleteModal = new bootstrap.Modal(deleteModalEl);
        setText("deleteFileName", btn.dataset.filename);
        setText("deleteClassName", btn.dataset.class || "-");
        const template = window.DATASET_DELETE_URL_TEMPLATE || "";
        deleteForm.action = template.replace("/0", "/" + btn.dataset.id);
        deleteModal.show();
    }

    document.querySelectorAll(".view-image-btn").forEach((btn) => {
        btn.addEventListener("click", function (e) {
            e.preventDefault();
            openViewModal(btn);
        });
    });

    document.querySelectorAll(".delete-image-btn").forEach((btn) => {
        btn.addEventListener("click", function (e) {
            e.preventDefault();
            openDeleteModal(btn);
        });
    });

    const reportBtn = document.getElementById("viewOptimizationReportBtn");
    const reportModalEl = document.getElementById("optimizationReportModal");
    if (reportBtn && reportModalEl) {
        reportBtn.addEventListener("click", function () {
            new bootstrap.Modal(reportModalEl).show();
        });
    }

    const splitForm = document.getElementById("splitDatasetForm");
    const splitSubmitBtn = document.getElementById("splitSubmitBtn");
    if (splitForm && splitSubmitBtn) {
        splitForm.addEventListener("submit", function () {
            splitSubmitBtn.disabled = true;
            splitSubmitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Memproses...';
        });
    }

    const optimizeForm = document.getElementById("optimizeDatasetForm");
    const optimizeSubmitBtn = document.getElementById("optimizeSubmitBtn");
    if (optimizeForm && optimizeSubmitBtn) {
        optimizeForm.addEventListener("submit", function () {
            optimizeSubmitBtn.disabled = true;
            optimizeSubmitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Memulai...';
        });
    }

    if (window.DATASET_OPTIMIZATION_RUNNING && window.DATASET_OPTIMIZATION_STATUS_URL) {
        startOptimizationPolling();
    }
});

function startOptimizationPolling() {
    const statusUrl = window.DATASET_OPTIMIZATION_STATUS_URL;
    const panel = document.getElementById("optimizationProgressPanel");
    const bar = document.getElementById("optimizationProgressBar");
    const messageEl = document.getElementById("optimizationProgressMessage");
    const percentEl = document.getElementById("optimizationProgressPercent");
    const countEl = document.getElementById("optimizationProgressCount");
    const optimizeBtns = [document.getElementById("optimizeImagesBtn"), document.getElementById("optimizeImagesBtnCard")];

    if (panel) panel.classList.remove("d-none");
    optimizeBtns.forEach((btn) => {
        if (btn) btn.disabled = true;
    });

    const poll = () => {
        fetch(statusUrl, { headers: { Accept: "application/json" } })
            .then((res) => res.json())
            .then((data) => {
                const status = data.status || {};
                const progress = status.progress || 0;
                const processed = status.processed || 0;
                const total = status.total || 0;

                if (bar) bar.style.width = progress + "%";
                if (percentEl) percentEl.textContent = progress + "%";
                if (messageEl) messageEl.textContent = status.message || "Optimizing images...";
                if (countEl) countEl.textContent = processed + " / " + total + " gambar";

                if (status.status === "running") {
                    setTimeout(poll, 2000);
                } else if (status.status === "completed") {
                    window.location.reload();
                } else if (status.status === "failed") {
                    if (messageEl) messageEl.textContent = status.message || "Optimasi gagal.";
                    optimizeBtns.forEach((btn) => {
                        if (btn) btn.disabled = false;
                    });
                }
            })
            .catch(() => setTimeout(poll, 3000));
    };

    poll();
}
