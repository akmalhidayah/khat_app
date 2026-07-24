/**
 * Evaluation page — error table filtering, pair review, detail modal.
 */
(function () {
    "use strict";

    const PAGE_SIZE = 10;
    const CLASS_LABELS = {
        naskhi: "Naskhi",
        diwani: "Diwani",
        diwani_jali: "Diwani Jali",
        tsuluts: "Tsuluts",
    };

    function initEvaluationPage() {
        const rows = window.EVAL_MISCLASSIFIED || [];
        const api = window.EVAL_API || {};
        const tbody = document.getElementById("miscTableBody");
        const paginationEl = document.getElementById("miscPagination");
        const resultCountEl = document.getElementById("miscResultCount");
        const activeFilterEl = document.getElementById("miscActiveFilterBadge");
        const emptyStateEl = document.getElementById("miscEmptyState");
        const tableWrapEl = document.getElementById("evaluationResultsTable");
        const applyBtn = document.getElementById("applyMiscFilters");
        const resetBtn = document.getElementById("resetMiscFilters");

        if (!tbody) return;

        let currentPage = 1;
        let filtered = rows.slice();
        let activeRow = null;
        let activePairKey = null;

        const filters = {
            trueClass: document.getElementById("filterTrueClass"),
            predClass: document.getElementById("filterPredClass"),
            errorType: document.getElementById("filterErrorType"),
            filename: document.getElementById("filterFilename"),
            highConf: document.getElementById("filterHighConf"),
        };

        function esc(text) {
            if (text == null) return "";
            return String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;");
        }

        function basename(path) {
            return (path || "").split("/").pop();
        }

        function classLabel(key) {
            return CLASS_LABELS[key] || (key || "").replace(/_/g, " ").replace(/\b\w/g, function (c) {
                return c.toUpperCase();
            });
        }

        function pairKey(trueClass, predClass) {
            return (trueClass || "") + "->" + (predClass || "");
        }

        function readFiltersFromUrl() {
            const params = new URLSearchParams(window.location.search);
            if (filters.trueClass && params.has("true_class")) {
                filters.trueClass.value = params.get("true_class") || "";
            }
            if (filters.predClass && params.has("predicted_class")) {
                filters.predClass.value = params.get("predicted_class") || "";
            }
            if (filters.errorType && params.has("error_type")) {
                filters.errorType.value = params.get("error_type") || "";
            }
            if (filters.filename && params.has("search")) {
                filters.filename.value = params.get("search") || "";
            }
            if (filters.highConf) {
                filters.highConf.checked = params.get("confidence_high") === "1";
            }
        }

        function syncUrlFromFilters() {
            const params = new URLSearchParams();
            if (filters.trueClass && filters.trueClass.value) {
                params.set("true_class", filters.trueClass.value);
            }
            if (filters.predClass && filters.predClass.value) {
                params.set("predicted_class", filters.predClass.value);
            }
            if (filters.errorType && filters.errorType.value) {
                params.set("error_type", filters.errorType.value);
            }
            if (filters.filename && filters.filename.value.trim()) {
                params.set("search", filters.filename.value.trim());
            }
            if (filters.highConf && filters.highConf.checked) {
                params.set("confidence_high", "1");
            }
            const qs = params.toString();
            const next = qs ? window.location.pathname + "?" + qs : window.location.pathname;
            window.history.replaceState({}, "", next);
        }

        function hasActiveFilters() {
            return !!(
                (filters.trueClass && filters.trueClass.value) ||
                (filters.predClass && filters.predClass.value) ||
                (filters.errorType && filters.errorType.value) ||
                (filters.filename && filters.filename.value.trim()) ||
                (filters.highConf && filters.highConf.checked)
            );
        }

        function updateResultCount() {
            if (!resultCountEl) return;
            const trueVal = filters.trueClass && filters.trueClass.value;
            const predVal = filters.predClass && filters.predClass.value;
            if (trueVal && predVal) {
                resultCountEl.textContent =
                    "Showing " +
                    filtered.length +
                    " error" +
                    (filtered.length === 1 ? "" : "s") +
                    " for " +
                    classLabel(trueVal) +
                    " → " +
                    classLabel(predVal);
            } else if (hasActiveFilters()) {
                resultCountEl.textContent =
                    "Showing " + filtered.length + " filtered error" + (filtered.length === 1 ? "" : "s");
            } else {
                resultCountEl.textContent =
                    "Showing all " + filtered.length + " prediction error" + (filtered.length === 1 ? "" : "s");
            }
        }

        function updateActiveFilterBadge() {
            if (!activeFilterEl) return;
            const trueVal = filters.trueClass && filters.trueClass.value;
            const predVal = filters.predClass && filters.predClass.value;
            if (trueVal && predVal) {
                activeFilterEl.innerHTML =
                    '<span class="eval-v2-active-filter-badge"><i class="bi bi-funnel-fill me-1"></i>Reviewing: ' +
                    esc(classLabel(trueVal)) +
                    " → " +
                    esc(classLabel(predVal)) +
                    '</span><button type="button" class="btn eval-v2-filter-clear-btn btn-sm" id="miscClearPairFilter">Clear</button>';
                activeFilterEl.classList.remove("d-none");
                const clearBtn = document.getElementById("miscClearPairFilter");
                if (clearBtn) {
                    clearBtn.addEventListener("click", resetFilters);
                }
            } else if (hasActiveFilters()) {
                activeFilterEl.innerHTML =
                    '<span class="eval-v2-active-filter-badge"><i class="bi bi-funnel-fill me-1"></i>Filters active</span>';
                activeFilterEl.classList.remove("d-none");
            } else {
                activeFilterEl.innerHTML = "";
                activeFilterEl.classList.add("d-none");
            }
        }

        function updateReviewButtonStates() {
            document.querySelectorAll(".btn-review-pair").forEach(function (btn) {
                const key = pairKey(btn.dataset.trueClass, btn.dataset.predictedClass);
                btn.classList.toggle("is-active", key === activePairKey);
                btn.setAttribute("aria-pressed", key === activePairKey ? "true" : "false");
            });
        }

        function updateEmptyState() {
            const isEmpty = filtered.length === 0;
            if (emptyStateEl) {
                emptyStateEl.classList.toggle("d-none", !isEmpty);
                emptyStateEl.hidden = !isEmpty;
            }
            if (tableWrapEl) {
                tableWrapEl.classList.toggle("d-none", isEmpty && rows.length > 0);
                tableWrapEl.hidden = isEmpty && rows.length > 0;
            }
            if (paginationEl) {
                paginationEl.classList.toggle("d-none", isEmpty);
                paginationEl.hidden = isEmpty;
            }
        }

        function renderThumb(row) {
            if (row.image_available && row.image_url) {
                return (
                    '<div class="eval-v2-thumb-wrap has-image">' +
                    '<img src="' +
                    esc(row.image_url) +
                    '" class="eval-v2-thumb evaluation-thumb" alt="Preview" loading="lazy" ' +
                    "onerror=\"this.style.display='none';var p=this.nextElementSibling;if(p)p.style.display='flex';\">" +
                    '<div class="eval-v2-thumb-placeholder" style="display:none;"><i class="bi bi-image-alt"></i><span>Unavailable</span></div>' +
                    "</div>"
                );
            }
            return (
                '<div class="eval-v2-thumb-wrap no-image">' +
                '<div class="eval-v2-thumb-placeholder"><i class="bi bi-image-alt"></i><span>Unavailable</span></div>' +
                "</div>"
            );
        }

        function render() {
            const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
            if (currentPage > totalPages) currentPage = totalPages;
            const start = (currentPage - 1) * PAGE_SIZE;
            const pageRows = filtered.slice(start, start + PAGE_SIZE);

            tbody.innerHTML = pageRows
                .map(function (row) {
                    const conf = row.confidence_pct != null ? row.confidence_pct : "—";
                    const margin = row.margin_pct != null ? row.margin_pct : "—";
                    const reviewed = row.reviewed_status && row.reviewed_status !== "pending";
                    return (
                        '<tr class="' +
                        (row.is_high_confidence_error ? "row-critical " : "") +
                        (reviewed ? "row-reviewed" : "") +
                        '">' +
                        "<td>" +
                        renderThumb(row) +
                        "</td>" +
                        '<td class="eval-v2-filename" title="' +
                        esc(row.filename || row.image_rel || "") +
                        '">' +
                        esc(basename(row.filename || row.image_rel)) +
                        (reviewed ? ' <span class="eval-v2-reviewed-badge">Reviewed</span>' : "") +
                        "</td>" +
                        '<td><span class="class-pill ' +
                        esc(row.true_label) +
                        '">' +
                        esc(row.true_display || row.true_label) +
                        "</span></td>" +
                        '<td><span class="class-pill ' +
                        esc(row.predicted_label) +
                        '">' +
                        esc(row.predicted_display || row.predicted_label) +
                        "</span></td>" +
                        "<td><strong>" +
                        conf +
                        "%</strong></td>" +
                        "<td>" +
                        margin +
                        "%</td>" +
                        '<td><span class="eval-v2-err-badge ' +
                        esc(row.error_badge_class || "err-label") +
                        '">' +
                        esc(row.error_type_label || row.error_type || "—") +
                        "</span></td>" +
                        '<td class="eval-v2-reco-cell">' +
                        esc(row.recommendation || row.error_hint || "—") +
                        "</td>" +
                        '<td><button type="button" class="btn eval-v2-view-btn misc-view-btn" data-id="' +
                        row.id +
                        '" title="View error detail" aria-label="View error detail"><i class="bi bi-eye"></i></button></td>' +
                        "</tr>"
                    );
                })
                .join("");

            if (paginationEl) {
                let html =
                    '<small class="text-muted">Page ' +
                    currentPage +
                    " / " +
                    totalPages +
                    " · " +
                    filtered.length +
                    " rows</small>";
                if (totalPages > 1) {
                    html +=
                        '<ul class="pagination pagination-sm mb-0">' +
                        '<li class="page-item ' +
                        (currentPage <= 1 ? "disabled" : "") +
                        '"><a class="page-link" href="#" data-page="' +
                        (currentPage - 1) +
                        '">Prev</a></li>' +
                        '<li class="page-item ' +
                        (currentPage >= totalPages ? "disabled" : "") +
                        '"><a class="page-link" href="#" data-page="' +
                        (currentPage + 1) +
                        '">Next</a></li></ul>';
                }
                paginationEl.innerHTML = html;
            }

            updateResultCount();
            updateActiveFilterBadge();
            updateEmptyState();
            updateReviewButtonStates();
        }

        function applyFilters(options) {
            options = options || {};
            const search = (filters.filename && filters.filename.value || "").trim().toLowerCase();
            filtered = rows.filter(function (row) {
                if (filters.trueClass && filters.trueClass.value && row.true_label !== filters.trueClass.value) {
                    return false;
                }
                if (filters.predClass && filters.predClass.value && row.predicted_label !== filters.predClass.value) {
                    return false;
                }
                if (filters.errorType && filters.errorType.value && row.error_type !== filters.errorType.value) {
                    return false;
                }
                if (filters.highConf && filters.highConf.checked && !row.is_high_confidence_error) {
                    return false;
                }
                if (search) {
                    const hay = ((row.filename || "") + " " + (row.image_rel || "")).toLowerCase();
                    if (!hay.includes(search)) return false;
                }
                return true;
            });

            const trueVal = filters.trueClass && filters.trueClass.value;
            const predVal = filters.predClass && filters.predClass.value;
            activePairKey = trueVal && predVal ? pairKey(trueVal, predVal) : null;

            currentPage = 1;
            syncUrlFromFilters();
            render();

            if (options.scrollToTable) {
                scrollToResultsTable();
            }
        }

        function resetFilters() {
            if (filters.trueClass) filters.trueClass.value = "";
            if (filters.predClass) filters.predClass.value = "";
            if (filters.errorType) filters.errorType.value = "";
            if (filters.filename) filters.filename.value = "";
            if (filters.highConf) filters.highConf.checked = false;
            activePairKey = null;
            window.history.replaceState({}, "", window.location.pathname);
            applyFilters();
        }

        function applyPairReviewFilter(trueClass, predClass, scroll) {
            if (filters.trueClass) filters.trueClass.value = trueClass || "";
            if (filters.predClass) filters.predClass.value = predClass || "";
            if (filters.errorType) filters.errorType.value = "";
            if (filters.filename) filters.filename.value = "";
            if (filters.highConf) filters.highConf.checked = false;
            activePairKey = pairKey(trueClass, predClass);
            applyFilters({ scrollToTable: scroll !== false });
        }

        function scrollToResultsTable() {
            const target = document.getElementById("evaluationResultsTable") || document.getElementById("misclassificationSection");
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }

        function setButtonLoading(btn, loading, loadingText) {
            if (!btn) return;
            if (loading) {
                btn.disabled = true;
                btn.dataset.originalHtml = btn.innerHTML;
                btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>' + (loadingText || "Loading...");
            } else {
                btn.disabled = false;
                if (btn.dataset.originalHtml) {
                    btn.innerHTML = btn.dataset.originalHtml;
                    delete btn.dataset.originalHtml;
                }
            }
        }

        function renderPreviewImage(row) {
            if (row.image_available && row.image_url) {
                return (
                    '<div class="eval-error-preview-frame">' +
                    '<img src="' +
                    esc(row.image_url) +
                    '" class="eval-error-preview-img evaluation-modal-image" alt="Error preview" ' +
                    "onerror=\"this.style.display='none';var p=this.nextElementSibling;if(p)p.style.display='flex';\">" +
                    '<div class="eval-error-preview-unavailable" style="display:none;"><i class="bi bi-image-alt"></i><strong>Image unavailable</strong><span>Preview file could not be loaded. Check stored image path.</span></div>' +
                    "</div>"
                );
            }
            return (
                '<div class="eval-error-preview-unavailable">' +
                '<i class="bi bi-image-alt"></i>' +
                "<strong>Image unavailable</strong>" +
                "<span>Preview file could not be loaded. Check stored image path.</span>" +
                "</div>"
            );
        }

        function openPreview(row) {
            if (!row) return;
            activeRow = row;
            const modalEl = document.getElementById("miscPreviewModal");
            const body = document.getElementById("miscPreviewBody");
            const title = document.getElementById("miscPreviewTitle");
            if (!modalEl || !body) return;

            if (title) {
                title.textContent = basename(row.filename || row.image_rel || "Prediction Error Detail");
            }

            const actions = (row.modal_actions || [])
                .map(function (item) {
                    return '<li><i class="bi bi-check2-circle me-1"></i>' + esc(item) + "</li>";
                })
                .join("");

            body.innerHTML =
                '<div class="eval-error-modal-grid">' +
                '<div class="eval-error-modal-col">' +
                '<div class="eval-error-card">' +
                '<h6 class="eval-error-card-title"><i class="bi bi-image me-1"></i>Image Preview</h6>' +
                renderPreviewImage(row) +
                '<div class="eval-error-meta-grid mt-3">' +
                "<div><span>Filename</span><strong>" +
                esc(row.filename || row.image_rel || "—") +
                "</strong></div>" +
                "<div><span>Dimensions</span><strong>" +
                (row.width && row.height ? row.width + " × " + row.height + " px" : "—") +
                "</strong></div>" +
                "<div><span>File Size</span><strong>" +
                esc(row.size_display || "—") +
                "</strong></div>" +
                "<div><span>Input Source</span><strong>" +
                esc(row.input_source || "Test Holdout Dataset") +
                "</strong></div>" +
                "<div><span>Preview Status</span><strong>" +
                esc(row.preview_status === "available" || row.image_available ? "Available" : "Unavailable") +
                "</strong></div>" +
                "<div><span>Image Source</span><strong>" +
                esc(row.source_folder || "—") +
                "</strong></div>" +
                "</div></div></div>" +
                '<div class="eval-error-modal-col">' +
                '<div class="eval-error-card">' +
                '<h6 class="eval-error-card-title"><i class="bi bi-clipboard-data me-1"></i>Error Summary</h6>' +
                '<div class="eval-error-summary-grid">' +
                "<div><span>True Class</span><strong><span class=\"class-pill " +
                esc(row.true_label) +
                '">' +
                esc(row.true_display) +
                "</span></strong></div>" +
                "<div><span>Predicted Class</span><strong><span class=\"class-pill " +
                esc(row.predicted_label) +
                '">' +
                esc(row.predicted_display) +
                "</span></strong></div>" +
                "<div><span>Confidence</span><strong>" +
                esc(row.confidence_pct) +
                "%</strong></div>" +
                "<div><span>Top-2 Margin</span><strong>" +
                esc(row.margin_pct) +
                "%</strong></div>" +
                "<div><span>Error Type</span><strong><span class=\"eval-v2-err-badge " +
                esc(row.error_badge_class) +
                '">' +
                esc(row.error_type_label) +
                "</span></strong></div>" +
                "<div><span>Recommendation</span><strong>" +
                esc(row.recommendation || row.error_hint) +
                "</strong></div>" +
                "</div></div>" +
                '<div class="eval-error-card mt-3">' +
                '<h6 class="eval-error-card-title"><i class="bi bi-search me-1"></i>Error Analysis</h6>' +
                '<p class="mb-0">' +
                esc(row.error_analysis || row.error_hint || "Manual review recommended.") +
                "</p></div>" +
                '<div class="eval-error-card mt-3">' +
                '<h6 class="eval-error-card-title"><i class="bi bi-list-check me-1"></i>Recommended Actions</h6>' +
                '<ul class="eval-error-action-list mb-0">' +
                actions +
                "</ul></div>" +
                '<details class="eval-error-details mt-3"><summary>Algorithm Calculation Detail</summary>' +
                '<div class="eval-error-algo-grid">' +
                "<div><span>Top-1 Class</span><strong>" +
                esc(row.predicted_display) +
                "</strong></div>" +
                "<div><span>Top-1 Score</span><strong>" +
                esc(row.confidence_pct) +
                "%</strong></div>" +
                "<div><span>Top-2 Class</span><strong>" +
                esc(row.top2_display || "—") +
                "</strong></div>" +
                "<div><span>Top-2 Score</span><strong>" +
                esc(row.top2_score_pct != null ? row.top2_score_pct + "%" : "—") +
                "</strong></div>" +
                '<div class="wide"><span>Margin Calculation</span><strong><code>' +
                esc(row.margin_calculation || "—") +
                "</code></strong></div>" +
                "<div><span>Confidence Category</span><strong>" +
                esc(row.confidence_category || "—") +
                "</strong></div>" +
                "<div><span>Validation Result</span><strong>Prediction Mismatch</strong></div>" +
                "</div></details>" +
                '<div class="eval-error-card mt-3">' +
                '<label class="form-label small" for="miscCorrectionNote">Correction Note</label>' +
                '<textarea class="form-control form-control-sm" id="miscCorrectionNote" rows="2" placeholder="Optional review note...">' +
                esc(row.correction_note || "") +
                "</textarea>" +
                '<label class="form-label small mt-2" for="miscCorrectionClass">Correct Class</label>' +
                '<select class="form-select form-select-sm" id="miscCorrectionClass">' +
                '<option value="">Select correct class</option>' +
                ["naskhi", "diwani", "diwani_jali", "tsuluts"]
                    .map(function (cls) {
                        const selected = cls === row.true_label ? " selected" : "";
                        return (
                            '<option value="' +
                            cls +
                            '"' +
                            selected +
                            ">" +
                            classLabel(cls) +
                            "</option>"
                        );
                    })
                    .join("") +
                "</select></div></div></div>";

            if (window.bootstrap && window.bootstrap.Modal) {
                window.bootstrap.Modal.getOrCreateInstance(modalEl).show();
            }
        }

        function postJson(url, payload) {
            return fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }).then(function (res) {
                return res.json();
            });
        }

        if (applyBtn) {
            applyBtn.addEventListener("click", function () {
                setButtonLoading(applyBtn, true, "Applying...");
                window.setTimeout(function () {
                    applyFilters({ scrollToTable: false });
                    setButtonLoading(applyBtn, false);
                }, 120);
            });
        }

        if (resetBtn) {
            resetBtn.addEventListener("click", resetFilters);
        }

        document.addEventListener("click", function (e) {
            const reviewBtn = e.target.closest(".btn-review-pair, .misc-filter-link");
            if (reviewBtn) {
                e.preventDefault();
                applyPairReviewFilter(reviewBtn.dataset.trueClass, reviewBtn.dataset.predictedClass, true);
                return;
            }

            const viewBtn = e.target.closest(".misc-view-btn");
            if (viewBtn) {
                e.preventDefault();
                const id = parseInt(viewBtn.dataset.id, 10);
                const row = rows.find(function (r) {
                    return r.id === id;
                });
                openPreview(row);
                return;
            }

            const pageLink = e.target.closest("#miscPagination [data-page]");
            if (pageLink) {
                e.preventDefault();
                const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
                const p = parseInt(pageLink.dataset.page, 10);
                if (p >= 1 && p <= totalPages) {
                    currentPage = p;
                    render();
                }
                return;
            }

            const emptyReset = e.target.closest("#miscEmptyResetBtn");
            if (emptyReset) {
                e.preventDefault();
                resetFilters();
            }
        });

        if (filters.filename) {
            filters.filename.addEventListener("keydown", function (e) {
                if (e.key === "Enter") {
                    e.preventDefault();
                    applyFilters({ scrollToTable: false });
                }
            });
        }

        const markReviewedBtn = document.getElementById("miscMarkReviewedBtn");
        if (markReviewedBtn) {
            markReviewedBtn.addEventListener("click", function () {
                if (!activeRow || !api.markReviewed) return;
                const note = document.getElementById("miscCorrectionNote");
                postJson(api.markReviewed, {
                    image_rel: activeRow.image_rel,
                    note: note ? note.value : "",
                }).then(function (data) {
                    if (data.ok) {
                        activeRow.reviewed_status = "reviewed";
                        render();
                        const modalEl = document.getElementById("miscPreviewModal");
                        if (modalEl && window.bootstrap) {
                            window.bootstrap.Modal.getInstance(modalEl)?.hide();
                        }
                    } else {
                        alert(data.message || "Could not mark as reviewed.");
                    }
                });
            });
        }

        function saveCorrection(action) {
            if (!activeRow || !api.saveCorrection) return;
            const clsEl = document.getElementById("miscCorrectionClass");
            const noteEl = document.getElementById("miscCorrectionNote");
            const correctClass = clsEl ? clsEl.value : activeRow.true_label;
            if (!correctClass) {
                alert("Please select the correct class.");
                return;
            }
            postJson(api.saveCorrection, {
                image_rel: activeRow.image_rel,
                correct_class: correctClass,
                note: noteEl ? noteEl.value : "",
                action: action,
            }).then(function (data) {
                if (data.ok) {
                    activeRow.reviewed_status = data.record.reviewed_status || "corrected";
                    activeRow.correction_label = data.record.correction_label;
                    render();
                    const modalEl = document.getElementById("miscPreviewModal");
                    if (modalEl && window.bootstrap) {
                        window.bootstrap.Modal.getInstance(modalEl)?.hide();
                    }
                } else {
                    alert(data.message || "Correction failed.");
                }
            });
        }

        ["miscSaveCorrectionBtn", "miscAddDatasetBtn"].forEach(function (id) {
            const btn = document.getElementById(id);
            if (btn) btn.addEventListener("click", function () {
                saveCorrection("save_to_dataset");
            });
        });

        const moveBtn = document.getElementById("miscMoveClassBtn");
        if (moveBtn) {
            moveBtn.addEventListener("click", function () {
                saveCorrection("move_to_class");
            });
        }

        readFiltersFromUrl();
        if (hasActiveFilters()) {
            applyFilters({ scrollToTable: false });
        } else {
            render();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initEvaluationPage);
    } else {
        initEvaluationPage();
    }
})();
