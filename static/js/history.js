/**
 * Classification History — filtering, pagination, sorting, modals.
 */
(function () {
  "use strict";

  const allRows = window.HISTORY_RECORDS || [];
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;

  let filtered = allRows.slice();
  let currentPage = 1;
  let pageSize = 10;
  let sortKey = "date";
  let sortDir = "desc";
  let quickFilter = "all";
  let activeRecord = null;

  const els = {
    countLabel: document.getElementById("historyCountLabel"),
    pageInfo: document.getElementById("pageInfo"),
    prevBtn: document.getElementById("prevPageBtn"),
    nextBtn: document.getElementById("nextPageBtn"),
    pageSizeSelect: document.getElementById("pageSizeSelect"),
    quickFilters: document.getElementById("quickFilters"),
    previewModal: document.getElementById("historyPreviewModal"),
    deleteModal: document.getElementById("historyDeleteModal"),
    deleteForm: document.getElementById("deleteRecordForm"),
    modalDeleteBtn: document.getElementById("modalDeleteBtn"),
    activeFilterNote: document.getElementById("activeFilterNote"),
    filteredEmpty: document.getElementById("historyFilteredEmpty"),
    tableWrap: document.querySelector(".hist-v2-table-wrap"),
    pagination: document.getElementById("historyPagination"),
  };

  const PLACEHOLDER =
    "data:image/svg+xml," +
    encodeURIComponent(
      '<svg xmlns="http://www.w3.org/2000/svg" width="56" height="56" rx="10" fill="#e2e8f0"><text x="50%" y="54%" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="sans-serif">N/A</text></svg>'
    );

  const QUICK_LABELS = {
    all: "Semua record",
    khat: "Khat Valid",
    non_khat: "Non-Khat Ditolak",
    uncertain: "Tidak Pasti",
    high: "High Confidence (≥85%)",
    medium: "Medium Confidence (50–84%)",
    low: "Low Confidence (<50%)",
  };

  function toneClass(tone) {
    return (
      {
        ok: "hist-v2-badge ok",
        warn: "hist-v2-badge warn",
        danger: "hist-v2-badge danger",
        info: "hist-v2-badge info",
      }[tone] || "hist-v2-badge neutral"
    );
  }

  function rowClass(row) {
    let cls = "hist-v2-row ";
    if (row.row_tone === "rejected") cls += "rejected";
    else if (row.row_tone === "uncertain") cls += "uncertain";
    else cls += "valid";
    if (row.is_valid_khat && row.confidence_band === "low") cls += " low-conf";
    return cls;
  }

  function confBarClass(band) {
    if (band === "high") return "bg-success";
    if (band === "medium") return "bg-warning";
    if (band === "low") return "bg-danger";
    return "bg-secondary";
  }

  function applyQuickFilter(rows) {
    if (quickFilter === "all") return rows;
    if (quickFilter === "khat") return rows.filter((r) => r.input_status === "khat");
    if (quickFilter === "non_khat") return rows.filter((r) => r.input_status === "non_khat");
    if (quickFilter === "uncertain") return rows.filter((r) => r.input_status === "uncertain");
    if (quickFilter === "high") {
      return rows.filter((r) => r.is_valid_khat && r.confidence_band === "high");
    }
    if (quickFilter === "medium") {
      return rows.filter((r) => r.is_valid_khat && r.confidence_band === "medium");
    }
    if (quickFilter === "low") {
      return rows.filter((r) => r.is_valid_khat && r.confidence_band === "low");
    }
    return rows;
  }

  function sortRows(rows) {
    const dir = sortDir === "asc" ? 1 : -1;
    return rows.slice().sort(function (a, b) {
      let va;
      let vb;
      switch (sortKey) {
        case "input_type":
          va = a.input_type_label || "";
          vb = b.input_type_label || "";
          break;
        case "status":
          va = a.status_label || "";
          vb = b.status_label || "";
          break;
        case "class":
          va = a.predicted_display || "";
          vb = b.predicted_display || "";
          break;
        case "confidence":
          va = a.confidence_pct != null ? a.confidence_pct : -1;
          vb = b.confidence_pct != null ? b.confidence_pct : -1;
          break;
        case "date":
        default:
          va = a.created_at_iso || "";
          vb = b.created_at_iso || "";
          break;
      }
      if (va < vb) return -1 * dir;
      if (va > vb) return 1 * dir;
      return 0;
    });
  }

  function updateSortHeaders() {
    document.querySelectorAll(".hist-v2-table th.sortable").forEach(function (th) {
      const key = th.getAttribute("data-sort");
      const icon = th.querySelector(".sort-icon");
      const isActive = key === sortKey;
      th.classList.toggle("sorted", isActive);
      th.classList.toggle("asc", isActive && sortDir === "asc");
      th.classList.toggle("desc", isActive && sortDir === "desc");
      if (icon) {
        icon.className =
          "bi sort-icon " +
          (isActive ? (sortDir === "asc" ? "bi-chevron-up" : "bi-chevron-down") : "bi-chevron-expand");
      }
    });
  }

  function updateActiveFilterNote(total) {
    if (!els.activeFilterNote) return;
    if (quickFilter === "all") return;
    const base = els.activeFilterNote.textContent.split("·")[0].trim();
    els.activeFilterNote.textContent = base + " · Chip: " + QUICK_LABELS[quickFilter] + " · " + total + " record";
  }

  function refreshFiltered() {
    filtered = sortRows(applyQuickFilter(allRows));
    currentPage = 1;
    render();
  }

  function render() {
    const total = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    if (currentPage > totalPages) currentPage = totalPages;
    const start = (currentPage - 1) * pageSize;
    const pageRows = filtered.slice(start, start + pageSize);
    const isEmpty = total === 0;

    if (els.countLabel) els.countLabel.textContent = total + " record ditampilkan";
    if (els.pageInfo) els.pageInfo.textContent = "Halaman " + currentPage + " / " + totalPages;
    if (els.prevBtn) els.prevBtn.disabled = currentPage <= 1 || isEmpty;
    if (els.nextBtn) els.nextBtn.disabled = currentPage >= totalPages || isEmpty;

    if (els.tableWrap) els.tableWrap.classList.toggle("d-none", isEmpty);
    if (els.pagination) els.pagination.classList.toggle("d-none", isEmpty);
    if (els.filteredEmpty) els.filteredEmpty.classList.toggle("d-none", !isEmpty);

    updateActiveFilterNote(total);

    if (isEmpty) {
      tbody.innerHTML = "";
      return;
    }

    tbody.innerHTML = pageRows
      .map(function (row) {
        const imgSrc = row.image_url || PLACEHOLDER;
        const inputTone =
          row.input_type_tone === "rejected" ? "danger" : row.input_type_tone === "uncertain" ? "warn" : "ok";
        const inputBadge = toneClass(inputTone);
        const statusBadge = toneClass(row.status_tone);
        const classHtml = row.predicted_class
          ? '<span class="class-pill ' + row.predicted_class + '">' + row.predicted_display + "</span>"
          : '<span class="text-muted">—</span>';

        let confHtml = '<span class="text-muted">—</span>';
        if (row.confidence_pct != null && row.is_valid_khat) {
          confHtml =
            '<div class="hist-v2-conf">' +
            '<div class="hist-v2-conf-pct">' +
            row.confidence_pct +
            "%</div>" +
            '<div class="progress hist-v2-progress"><div class="progress-bar ' +
            confBarClass(row.confidence_band) +
            '" style="width:' +
            Math.min(row.confidence_pct, 100) +
            '%"></div></div>' +
            '<small class="text-muted">' +
            row.confidence_label +
            "</small></div>";
        }

        const khatLine =
          row.khat_probability_pct != null && row.input_status !== "khat"
            ? '<div class="hist-v2-sub">Khat ' + row.khat_probability_pct + "%</div>"
            : "";

        const fname = row.filename || "—";
        const fnameShort = fname.length > 22 ? fname.slice(0, 20) + "…" : fname;

        return (
          '<tr class="' +
          rowClass(row) +
          '" data-id="' +
          row.id +
          '">' +
          '<td><button type="button" class="hist-v2-thumb-btn view-btn" title="Lihat pratinjau" data-id="' +
          row.id +
          '"><img src="' +
          imgSrc +
          '" class="hist-v2-thumb" alt="" loading="lazy" onerror="this.src=\'' +
          PLACEHOLDER +
          "'\"></button></td>" +
          '<td><span class="' +
          inputBadge +
          '">' +
          row.input_type_label +
          "</span>" +
          khatLine +
          "</td>" +
          '<td><span class="' +
          statusBadge +
          '">' +
          row.status_label +
          "</span>" +
          (row.manual_review_required ? '<div class="hist-v2-sub warn">Review manual</div>' : "") +
          "</td>" +
          "<td>" +
          classHtml +
          "</td>" +
          "<td>" +
          confHtml +
          "</td>" +
          '<td><div class="hist-v2-date-main">' +
          row.created_at +
          '</div><small class="hist-v2-filename" title="' +
          fname.replace(/"/g, "&quot;") +
          '">' +
          fnameShort +
          "</small></td>" +
          '<td><div class="hist-v2-actions">' +
          '<a href="/classification/detail/' +
          row.id +
          '" class="btn btn-sm btn-outline-teal" title="Detail perhitungan"><i class="bi bi-calculator"></i></a>' +
          '<button type="button" class="btn btn-sm btn-outline-primary view-btn" title="Lihat pratinjau" data-id="' +
          row.id +
          '"><i class="bi bi-eye"></i></button>' +
          '<button type="button" class="btn btn-sm btn-outline-danger delete-btn" title="Hapus riwayat" data-id="' +
          row.id +
          '"><i class="bi bi-trash"></i></button>' +
          "</div></td></tr>"
        );
      })
      .join("");

    bindRowActions();
  }

  function findRecord(id) {
    return allRows.find(function (r) {
      return String(r.id) === String(id);
    });
  }

  function appendQueryToDeleteForm(url) {
    if (!els.deleteForm) return url;
    const qs = window.HISTORY_QUERY_STRING;
    if (qs) {
      els.deleteForm.action = url + "?" + qs;
    } else {
      els.deleteForm.action = url;
    }
  }

  function renderClassPill(classKey, display) {
    if (!classKey) {
      return '<span class="hist-detail-chip neutral">—</span>';
    }
    return (
      '<span class="class-pill ' +
      classKey +
      ' hist-detail-class-pill">' +
      display +
      "</span>"
    );
  }

  function renderStatusChip(tone, label) {
    return (
      '<span class="hist-v2-badge hist-detail-chip ' +
      (tone || "neutral") +
      '">' +
      label +
      "</span>"
    );
  }

  function renderConfidenceHighlight(row) {
    if (row.confidence_pct == null || !row.is_valid_khat) {
      return '<span class="hist-detail-chip neutral">—</span>';
    }
    const band = row.confidence_band || "neutral";
    const chipClass = band === "high" ? "ok" : band === "medium" ? "warn" : band === "low" ? "danger" : "neutral";
    let html =
      '<span class="hist-detail-chip ' +
      chipClass +
      '">' +
      row.confidence_pct +
      "%</span>";
    if (row.confidence_label) {
      html += '<small class="hist-detail-chip-note">' + row.confidence_label + "</small>";
    }
    return html;
  }

  function renderReliabilityHighlight(row) {
    const level = row.reliability_level;
    if (!level || level === "—") {
      return '<span class="hist-detail-chip neutral">—</span>';
    }
    let chipClass = "info";
    const lower = String(level).toLowerCase();
    if (lower.indexOf("tinggi") >= 0 || lower.indexOf("strong") >= 0 || lower.indexOf("high") >= 0) {
      chipClass = "ok";
    } else if (lower.indexOf("sedang") >= 0 || lower.indexOf("moderate") >= 0 || lower.indexOf("medium") >= 0) {
      chipClass = "warn";
    } else if (lower.indexOf("rendah") >= 0 || lower.indexOf("low") >= 0) {
      chipClass = "danger";
    }
    return '<span class="hist-detail-chip ' + chipClass + '">' + level + "</span>";
  }

  function setHtml(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  }

  function openPreview(row) {
    activeRecord = row;
    const detailBtn = document.getElementById("modalDetailCalcBtn");
    if (detailBtn && row.id) {
      detailBtn.href = "/classification/detail/" + row.id;
      detailBtn.classList.remove("d-none");
    }
    const img = document.getElementById("modalPreviewImg");
    const ph = document.getElementById("modalPreviewPlaceholder");
    const rejectNote = document.getElementById("modalRejectNote");
    const uncertainNote = document.getElementById("modalUncertainNote");
    const confBarWrap = document.getElementById("modalConfidenceBarWrap");
    const confBar = document.getElementById("modalConfidenceBar");

    if (img && row.image_url) {
      img.onerror = function () {
        img.classList.add("d-none");
        if (ph) ph.classList.remove("d-none");
      };
      img.src = row.image_url;
      img.classList.remove("d-none");
      if (ph) ph.classList.add("d-none");
    } else if (img) {
      img.classList.add("d-none");
      if (ph) ph.classList.remove("d-none");
    }

    const fname = row.filename || "—";
    setText("modalFilename", fname);
    setText("modalFilenamePreview", fname);
    const fnamePreview = document.getElementById("modalFilenamePreview");
    if (fnamePreview) fnamePreview.title = fname;

    setText("modalInputType", row.input_type_label);
    setText("modalStatus", row.status_label);
    setText("modalKhatProb", row.khat_probability_pct != null ? row.khat_probability_pct + "%" : "—");
    setText("modalNonKhatProb", row.non_khat_probability_pct != null ? row.non_khat_probability_pct + "%" : "—");
    setText("modalClass", row.predicted_class ? row.predicted_display : "—");
    setText("modalConfidence", row.confidence_pct != null ? row.confidence_pct + "%" : "—");
    setText("modalReliability", row.reliability_level || "—");
    setText("modalDate", row.created_at);
    setText("modalArch", row.model_architecture || "—");
    setText("modalMode", row.model_training_mode || "—");
    setText("modalManualExpected", row.manual_expected_display || row.manual_expected_class || "Unknown");
    setText("modalKnownClassStatus", row.known_class_status || "—");
    setText("modalPreprocessingMode", row.preprocessing_mode === "manuscript" ? "Manuscript Mode" : "Standard");
    setText("modalSourceType", row.source_type || "—");
    setText("modalCorrectionLabel", row.correction_label ? row.correction_label.replace(/_/g, " ") : "—");
    setText("modalCorrectionNotes", row.correction_notes || "—");

    setHtml(
      "modalHighlightClass",
      renderClassPill(row.predicted_class, row.predicted_display)
    );
    setHtml("modalHighlightConfidence", renderConfidenceHighlight(row));
    setHtml("modalHighlightReliability", renderReliabilityHighlight(row));
    setHtml("modalHighlightStatus", renderStatusChip(row.status_tone, row.status_label));

    if (confBarWrap && confBar) {
      if (row.confidence_pct != null && row.is_valid_khat) {
        confBarWrap.classList.remove("d-none");
        confBar.style.width = Math.min(row.confidence_pct, 100) + "%";
        confBar.className = "hist-detail-conf-bar-fill " + (row.confidence_band || "neutral");
      } else {
        confBarWrap.classList.add("d-none");
        confBar.style.width = "0%";
      }
    }

    if (rejectNote) rejectNote.classList.toggle("d-none", row.input_status !== "non_khat");
    if (uncertainNote) {
      uncertainNote.classList.toggle(
        "d-none",
        row.input_status !== "uncertain" && !row.manual_review_required
      );
    }

    const simSection = document.getElementById("modalSimilaritySection");
    const simCard = document.getElementById("modalSimilarityCard");
    const simMsg = document.getElementById("modalSimilarityMessage");
    const simAcademic = document.getElementById("modalSimilarityAcademic");
    if (row.similarity_available) {
      if (simSection) simSection.classList.remove("d-none");
      if (simCard) {
        simCard.className =
          "hist-detail-similarity-card similarity-tone-" + (row.similarity_tone || "neutral");
      }
      setText("modalSimilarityStatus", row.similarity_status || "—");
      setText(
        "modalSimilarityScore",
        row.similarity_score_pct != null ? row.similarity_score_pct + "%" : "—"
      );
      setText("modalNearestImage", row.nearest_dataset_image || "—");
      setText("modalNearestClass", row.nearest_dataset_class || "—");
      setText("modalSimilarityRisk", row.similarity_risk_level || "—");
      setText("modalSimilarityRecommendation", row.similarity_recommendation || "—");
      if (simMsg) {
        if (row.similarity_message) {
          simMsg.textContent = row.similarity_message;
          simMsg.classList.remove("d-none");
        } else {
          simMsg.textContent = "";
          simMsg.classList.add("d-none");
        }
      }
      if (simAcademic) {
        if (row.similarity_score != null && row.similarity_score >= 85) {
          simAcademic.textContent =
            "Untuk evaluasi akademik, sebaiknya gunakan gambar uji yang tidak pernah masuk dataset training. Jika gambar terlalu mirip dengan data training, hasil akurasi dapat terlihat tinggi tetapi tidak mencerminkan kemampuan generalisasi model.";
          simAcademic.classList.remove("d-none");
        } else {
          simAcademic.textContent = "";
          simAcademic.classList.add("d-none");
        }
      }
    } else if (simSection) {
      simSection.classList.add("d-none");
    }

    const charsSection = document.getElementById("modalCharacteristicsSection");
    const charsEl = document.getElementById("modalCharacteristics");
    if (charsEl) {
      if (row.characteristics && row.characteristics.predicted && window.KhatCharacteristics) {
        charsEl.innerHTML = window.KhatCharacteristics.renderHistoryCharacteristics(row.characteristics);
        charsEl.classList.remove("d-none");
        if (charsSection) charsSection.classList.remove("d-none");
      } else {
        charsEl.innerHTML = "";
        charsEl.classList.add("d-none");
        if (charsSection) charsSection.classList.add("d-none");
      }
    }

    const algoEl = document.getElementById("modalAlgorithmCalc");
    const algoSection = document.getElementById("modalAlgorithmSection");
    const algoCollapse = document.getElementById("modalAlgorithmCollapse");
    if (algoEl && window.AlgorithmCalculationRender) {
      algoEl.innerHTML = window.AlgorithmCalculationRender.render(row.algorithm_calculation);
      if (algoSection) algoSection.classList.remove("d-none");
      if (algoCollapse && window.bootstrap) {
        bootstrap.Collapse.getOrCreateInstance(algoCollapse, { toggle: false }).hide();
      }
    } else if (algoSection) {
      algoSection.classList.add("d-none");
    }

    if (els.previewModal && window.bootstrap) {
      bootstrap.Modal.getOrCreateInstance(els.previewModal).show();
    }
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function openDelete(row) {
    activeRecord = row;
    if (row.delete_url) appendQueryToDeleteForm(row.delete_url);
    if (els.deleteModal && window.bootstrap) {
      bootstrap.Modal.getOrCreateInstance(els.deleteModal).show();
    }
  }

  function bindRowActions() {
    tbody.querySelectorAll(".view-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const row = findRecord(btn.getAttribute("data-id"));
        if (row) openPreview(row);
      });
    });
    tbody.querySelectorAll(".delete-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const row = findRecord(btn.getAttribute("data-id"));
        if (row) openDelete(row);
      });
    });
  }

  if (els.quickFilters) {
    els.quickFilters.addEventListener("click", function (e) {
      const chip = e.target.closest("[data-quick]");
      if (!chip) return;
      quickFilter = chip.getAttribute("data-quick");
      els.quickFilters.querySelectorAll(".hist-v2-chip").forEach(function (c) {
        c.classList.toggle("active", c === chip);
      });
      refreshFiltered();
    });
  }

  if (els.pageSizeSelect) {
    els.pageSizeSelect.addEventListener("change", function () {
      pageSize = parseInt(this.value, 10) || 10;
      currentPage = 1;
      render();
    });
  }

  if (els.prevBtn) {
    els.prevBtn.addEventListener("click", function () {
      if (currentPage > 1) {
        currentPage -= 1;
        render();
      }
    });
  }

  if (els.nextBtn) {
    els.nextBtn.addEventListener("click", function () {
      const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
      if (currentPage < totalPages) {
        currentPage += 1;
        render();
      }
    });
  }

  document.querySelectorAll(".hist-v2-table th.sortable").forEach(function (th) {
    th.addEventListener("click", function () {
      const key = th.getAttribute("data-sort");
      if (sortKey === key) {
        sortDir = sortDir === "asc" ? "desc" : "asc";
      } else {
        sortKey = key;
        sortDir = key === "date" ? "desc" : "asc";
      }
      filtered = sortRows(filtered);
      updateSortHeaders();
      render();
    });
  });

  if (els.modalDeleteBtn) {
    els.modalDeleteBtn.addEventListener("click", function () {
      if (!activeRecord) return;
      if (els.previewModal && window.bootstrap) {
        const inst = bootstrap.Modal.getInstance(els.previewModal);
        if (inst) inst.hide();
      }
      openDelete(activeRecord);
    });
  }

  updateSortHeaders();
  refreshFiltered();
})();
