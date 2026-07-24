document.addEventListener("DOMContentLoaded", function () {
    const data = window.TRAINING_DATA || {};
    const form = document.getElementById("trainingForm");
    const confirmBtn = document.getElementById("confirmTrainingBtn");
    const confirmModal = document.getElementById("trainingConfirmModal");
    const processForm = document.getElementById("processDatasetForm");
    const processBtn = document.getElementById("processDatasetBtn");
    const uploadForm = document.getElementById("tmUploadForm");
    const uploadBtn = document.getElementById("tmUploadBtn");

    initUploadTabs();
    initTmUploadUi();
    initVersionToggle();
    initUploadSummary();
    initPathCopyButtons();

    const REQUIRED_LABELS = ["diwani", "diwani_jali", "naskhi", "tsuluts"];
    const LABEL_DISPLAY = {
        diwani: "diwani",
        diwani_jali: "diwani jali",
        naskhi: "naskhi",
        tsuluts: "tsuluts",
    };

    function normalizeTmLabel(raw) {
        const text = (raw || "").trim().toLowerCase().replace(/-/g, "_");
        const compact = text.replace(/\s+/g, "_");
        if (compact === "diwani_jali" || compact === "diwanijali" || (compact.includes("diwani") && compact.includes("jali"))) {
            return "diwani_jali";
        }
        if (REQUIRED_LABELS.includes(compact)) return compact;
        if (compact.includes("naskhi") || compact.includes("naski")) return "naskhi";
        if (compact.includes("diwani")) return "diwani";
        if (["tsuluts", "tsuluth", "thuluth", "sulus"].some((t) => compact.includes(t))) return "tsuluts";
        return compact;
    }

    function setCheckState(key, state) {
        const item = document.querySelector(`.mm-v2-validation-checks li[data-check="${key}"]`);
        if (!item) return;
        item.classList.remove("is-ok", "is-err", "is-pending");
        const icon = item.querySelector("i");
        if (state === "ok") {
            item.classList.add("is-ok");
            if (icon) icon.className = "bi bi-check-circle-fill";
        } else if (state === "err") {
            item.classList.add("is-err");
            if (icon) icon.className = "bi bi-x-circle-fill";
        } else {
            item.classList.add("is-pending");
            if (icon) icon.className = "bi bi-circle";
        }
    }

    function resetUploadSummary() {
        ["model_json", "metadata_json", "weights_bin", "labels", "input_size"].forEach((k) => setCheckState(k, "pending"));
    }

    function validateMetadataContent(metadata) {
        const rawLabels = metadata.labels || [];
        const normalized = rawLabels.map(normalizeTmLabel);
        const missing = REQUIRED_LABELS.filter((l) => !normalized.includes(l));
        const labelsOk = missing.length === 0 && rawLabels.length > 0;
        const imageSize = parseInt(metadata.imageSize || 224, 10);
        const sizeOk = imageSize === 224;
        return { labelsOk, sizeOk, missing, imageSize };
    }

    function updateUploadSummaryFromFiles() {
        const activePanel = document.querySelector("[data-upload-panel].active");
        const mode = activePanel ? activePanel.getAttribute("data-upload-panel") : "files";

        if (mode === "zip") {
            const zip = document.getElementById("tmZipInput");
            const hasZip = zip && zip.files.length && zip.files[0].name.toLowerCase().endsWith(".zip");
            setCheckState("model_json", hasZip ? "ok" : "pending");
            setCheckState("metadata_json", hasZip ? "ok" : "pending");
            setCheckState("weights_bin", hasZip ? "ok" : "pending");
            setCheckState("labels", hasZip ? "pending" : "pending");
            setCheckState("input_size", hasZip ? "pending" : "pending");
            return;
        }

        const files = {};
        document.querySelectorAll(".tm-file-input").forEach((input) => {
            const key = input.getAttribute("name");
            if (input.files.length) files[key] = input.files[0];
        });

        setCheckState("model_json", files.model_json ? "ok" : "pending");
        setCheckState("weights_bin", files.weights_bin ? "ok" : "pending");

        if (!files.metadata_json) {
            setCheckState("metadata_json", "pending");
            setCheckState("labels", "pending");
            setCheckState("input_size", "pending");
            return;
        }

        setCheckState("metadata_json", "ok");
        const reader = new FileReader();
        reader.onload = function () {
            try {
                const meta = JSON.parse(reader.result);
                const result = validateMetadataContent(meta);
                setCheckState("labels", result.labelsOk ? "ok" : "err");
                setCheckState("input_size", result.sizeOk ? "ok" : "err");
                if (!result.labelsOk) {
                    const labelsItem = document.querySelector('.mm-v2-validation-checks li[data-check="labels"] span');
                    if (labelsItem && result.missing.length) {
                        const missingDisplay = result.missing.map((l) => LABEL_DISPLAY[l] || l).join(", ");
                        labelsItem.textContent = `Label kurang: ${missingDisplay}`;
                    }
                } else {
                    const labelsItem = document.querySelector('.mm-v2-validation-checks li[data-check="labels"] span');
                    if (labelsItem) labelsItem.textContent = "Labels: diwani, diwani jali, naskhi, tsuluts";
                }
                if (!result.sizeOk) {
                    const sizeItem = document.querySelector('.mm-v2-validation-checks li[data-check="input_size"] span');
                    if (sizeItem) sizeItem.textContent = `Input size: ${result.imageSize} px (harus 224)`;
                } else {
                    const sizeItem = document.querySelector('.mm-v2-validation-checks li[data-check="input_size"] span');
                    if (sizeItem) sizeItem.textContent = "Input size: 224 × 224 px";
                }
            } catch (_err) {
                setCheckState("metadata_json", "err");
                setCheckState("labels", "pending");
                setCheckState("input_size", "pending");
            }
        };
        reader.readAsText(files.metadata_json);
    }

    function initUploadSummary() {
        resetUploadSummary();
    }

    const fields = {
        epochs: {
            el: document.getElementById("epochs"),
            validate() {
                if (!this.el) return true;
                const v = parseInt(this.el.value, 10);
                return !(isNaN(v) || v < 1 || v > 100);
            },
        },
        batch_size: {
            el: document.getElementById("batch_size"),
            validate() {
                if (!this.el) return true;
                const v = parseInt(this.el.value, 10);
                return !(isNaN(v) || v < 1 || v > 128);
            },
        },
        learning_rate: {
            el: document.getElementById("learning_rate"),
            validate() {
                if (!this.el) return true;
                const raw = this.el.value.trim().replace(",", ".");
                const v = parseFloat(raw);
                return !(isNaN(v) || v < 0.000001 || v > 1);
            },
        },
    };

    function validateForm() {
        return Object.values(fields).every((f) => f.validate());
    }

    function setRunningUi() {
        document.querySelectorAll("#startTrainingBtnPrimary, #startTrainingBtn").forEach((btn) => {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training Eksperimental...';
            }
        });
    }

    function openConfirmModal() {
        if (!form || !data.datasetReady || !data.canRetrain) return;
        if (data.buttonMode === "running") return;
        if (!validateForm()) return;
        new bootstrap.Modal(confirmModal).show();
    }

    const startBtn = document.getElementById("startTrainingBtnPrimary");
    if (startBtn) startBtn.addEventListener("click", openConfirmModal);

    if (confirmBtn && form) {
        confirmBtn.addEventListener("click", function () {
            if (!validateForm()) return;
            confirmBtn.disabled = true;
            confirmBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Memulai...';
            setRunningUi();
            bootstrap.Modal.getInstance(confirmModal)?.hide();
            form.submit();
        });
    }

    if (processForm && processBtn) {
        processForm.addEventListener("submit", function () {
            processBtn.disabled = true;
            processBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Memproses...';
        });
    }

    if (uploadForm && uploadBtn) {
        uploadForm.addEventListener("submit", function (e) {
            const activePanel = document.querySelector("[data-upload-panel].active");
            const mode = activePanel ? activePanel.getAttribute("data-upload-panel") : "files";
            if (mode === "files" && !validateSeparateFiles()) {
                e.preventDefault();
                return;
            }
            if (mode === "zip") {
                const zip = document.getElementById("tmZipInput");
                if (!zip || !zip.files.length) {
                    e.preventDefault();
                    setZipStatus("err", "Pilih arsip ZIP terlebih dahulu.");
                    return;
                }
                if (!zip.files[0].name.toLowerCase().endsWith(".zip")) {
                    e.preventDefault();
                    setZipStatus("err", "File harus berformat .zip");
                    return;
                }
            }
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Mengunggah...';
        });
    }

    function pollTrainingStatus() {
        if (!data.statusUrl) return;
        fetch(data.statusUrl, { headers: { Accept: "application/json" }, credentials: "same-origin" })
            .then((res) => res.json())
            .then((payload) => {
                if (payload.status === "running") {
                    setRunningUi();
                    return;
                }
                if (payload.button_mode === "retrain" || payload.status === "completed" || payload.status === "failed") {
                    window.location.reload();
                }
            })
            .catch(() => {});
    }

    if (data.isTrainingRunning || data.buttonMode === "running") {
        setInterval(pollTrainingStatus, 2000);
    }

    function initUploadTabs() {
        const tabs = document.querySelectorAll("[data-upload-tab]");
        const panels = document.querySelectorAll("[data-upload-panel]");
        if (!tabs.length) return;

        tabs.forEach((tab) => {
            tab.addEventListener("click", function () {
                const target = tab.getAttribute("data-upload-tab");
                tabs.forEach((t) => t.classList.toggle("active", t === tab));
                panels.forEach((panel) => {
                    panel.classList.toggle("active", panel.getAttribute("data-upload-panel") === target);
                });
                resetUploadSummary();
                updateUploadSummaryFromFiles();
            });
        });
    }

    function formatSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    }

    function validateFileName(file, expected) {
        if (!file) return { ok: false, message: "File belum dipilih" };
        const name = file.name.toLowerCase();
        if (expected === "model.json" && !name.endsWith(".json")) {
            return { ok: false, message: "Harus file .json" };
        }
        if (expected === "metadata.json" && !name.endsWith(".json")) {
            return { ok: false, message: "Harus file .json" };
        }
        if (expected === "weights.bin" && !name.endsWith(".bin")) {
            return { ok: false, message: "Harus file .bin" };
        }
        if (name !== expected && !name.endsWith("/" + expected)) {
            return { ok: true, message: "Nama file berbeda, akan divalidasi server" };
        }
        return { ok: true, message: "File valid" };
    }

    function setFileDropState(drop, state, statusText) {
        if (!drop) return;
        drop.classList.remove("is-valid", "is-invalid");
        if (state === "ok") drop.classList.add("is-valid");
        if (state === "err") drop.classList.add("is-invalid");
        const statusEl = drop.querySelector(".mm-v2-file-status");
        if (statusEl) {
            statusEl.className = "mm-v2-file-status " + (state === "ok" ? "ok" : state === "err" ? "err" : "");
            statusEl.textContent = statusText || "";
        }
    }

    function setZipStatus(state, message) {
        const drop = document.getElementById("tmZipDrop");
        if (!drop) return;
        drop.classList.remove("is-valid", "is-invalid");
        if (state === "ok") drop.classList.add("is-valid");
        if (state === "err") drop.classList.add("is-invalid");
        const statusEl = document.getElementById("tmZipStatus");
        if (statusEl) {
            statusEl.className = "mm-v2-file-status " + (state === "ok" ? "ok" : state === "err" ? "err" : "");
            statusEl.textContent = message || "";
        }
    }

    function validateSeparateFiles() {
        let allOk = true;
        document.querySelectorAll(".tm-file-input").forEach((input) => {
            const expected = input.getAttribute("data-expected");
            const drop = input.closest(".mm-v2-file-drop");
            if (!input.files.length) {
                setFileDropState(drop, "err", "Wajib diisi");
                allOk = false;
                return;
            }
            const result = validateFileName(input.files[0], expected);
            setFileDropState(drop, result.ok ? "ok" : "err", result.message);
            if (!result.ok) allOk = false;
        });

        const metaInput = document.querySelector('.tm-file-input[name="metadata_json"]');
        if (metaInput && metaInput.files.length) {
            const labelsCheck = document.querySelector('.mm-v2-validation-checks li[data-check="labels"]');
            const sizeCheck = document.querySelector('.mm-v2-validation-checks li[data-check="input_size"]');
            if (labelsCheck && labelsCheck.classList.contains("is-err")) allOk = false;
            if (sizeCheck && sizeCheck.classList.contains("is-err")) allOk = false;
        }

        return allOk;
    }

    function initTmUploadUi() {
        document.querySelectorAll(".tm-file-input").forEach((input) => {
            input.addEventListener("change", function () {
                const drop = input.closest(".mm-v2-file-drop");
                const nameEl = drop?.querySelector(".mm-v2-file-name");
                const metaEl = drop?.querySelector(".mm-v2-file-meta");
                const expected = input.getAttribute("data-expected");
                const file = input.files[0];
                if (!file) {
                    if (nameEl) nameEl.textContent = expected;
                    if (metaEl) metaEl.textContent = "Belum dipilih";
                    setFileDropState(drop, "", "");
                    return;
                }
                if (nameEl) nameEl.textContent = file.name;
                if (metaEl) metaEl.textContent = formatSize(file.size);
                const result = validateFileName(file, expected);
                setFileDropState(drop, result.ok ? "ok" : "err", result.ok ? result.message : result.message);
                updateUploadSummaryFromFiles();
            });
        });

        const zipInput = document.getElementById("tmZipInput");
        const zipLabel = document.getElementById("tmZipLabel");
        const zipMeta = document.getElementById("tmZipMeta");
        const zipDrop = document.getElementById("tmZipDrop");

        function handleZip(file) {
            if (!file) {
                if (zipLabel) zipLabel.textContent = "Seret arsip ZIP atau klik untuk memilih";
                if (zipMeta) zipMeta.textContent = "";
                setZipStatus("", "");
                return;
            }
            if (zipLabel) zipLabel.textContent = file.name;
            if (zipMeta) zipMeta.textContent = formatSize(file.size);
            if (file.name.toLowerCase().endsWith(".zip")) {
                setZipStatus("ok", "Arsip ZIP siap diunggah");
            } else {
                setZipStatus("err", "File harus berformat .zip");
            }
            updateUploadSummaryFromFiles();
        }

        if (zipInput) {
            zipInput.addEventListener("change", function () {
                handleZip(zipInput.files[0]);
            });
        }
        if (zipDrop && zipInput) {
            ["dragenter", "dragover"].forEach((evt) => {
                zipDrop.addEventListener(evt, function (e) {
                    e.preventDefault();
                    zipDrop.classList.add("dragover");
                });
            });
            ["dragleave", "drop"].forEach((evt) => {
                zipDrop.addEventListener(evt, function (e) {
                    e.preventDefault();
                    zipDrop.classList.remove("dragover");
                });
            });
            zipDrop.addEventListener("drop", function (e) {
                const file = e.dataTransfer?.files?.[0];
                if (file) {
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    zipInput.files = dt.files;
                    handleZip(file);
                }
            });
        }
    }

    function initVersionToggle() {
        const btn = document.getElementById("toggleAllVersions");
        const scroll = document.querySelector(".mm-v2-version-scroll");
        if (!btn || !scroll) return;

        btn.addEventListener("click", function () {
            const expanded = btn.getAttribute("data-expanded") === "true";
            const next = !expanded;
            scroll.classList.toggle("show-all", next);
            btn.setAttribute("data-expanded", next ? "true" : "false");
            btn.textContent = next
                ? btn.getAttribute("data-label-hide") || "Sembunyikan Versi Lama"
                : btn.getAttribute("data-label-show") || "Lihat Semua Versi";
        });
    }

    function initPathCopyButtons() {
        document.querySelectorAll(".mm-v2-copy-btn[data-copy]").forEach(function (btn) {
            btn.addEventListener("click", function () {
                const text = btn.getAttribute("data-copy") || "";
                if (!text) return;

                function onCopied() {
                    btn.classList.add("is-copied");
                    const icon = btn.querySelector("i");
                    if (icon) icon.className = "bi bi-check2";
                    setTimeout(function () {
                        btn.classList.remove("is-copied");
                        if (icon) icon.className = "bi bi-clipboard";
                    }, 1600);
                }

                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(text).then(onCopied).catch(function () {
                        fallbackCopy(text, onCopied);
                    });
                } else {
                    fallbackCopy(text, onCopied);
                }
            });
        });

        function fallbackCopy(text, onSuccess) {
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.setAttribute("readonly", "");
            ta.style.position = "fixed";
            ta.style.left = "-9999px";
            document.body.appendChild(ta);
            ta.select();
            try {
                document.execCommand("copy");
                onSuccess();
            } catch (_err) {
                /* ignore */
            }
            document.body.removeChild(ta);
        }
    }
});
