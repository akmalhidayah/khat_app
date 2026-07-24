/**
 * Render Detail Perhitungan Algoritma from JSON (history modal).
 */
(function (global) {
  "use strict";

  function esc(text) {
    if (text == null) return "";
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderSummaryGrid(items) {
    if (!items || !items.length) return "";
    return (
      '<div class="algorithm-summary-grid">' +
      items
        .map(function (item) {
          return (
            '<div class="algorithm-summary-mini-card"><span class="algorithm-summary-label">' +
            esc(item.label) +
            '</span><strong class="algorithm-summary-value">' +
            esc(item.value) +
            "</strong></div>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function renderPipeline(steps) {
    if (!steps || !steps.length) return "";
    return (
      '<div class="algorithm-pipeline">' +
      steps
        .map(function (step) {
          return (
            '<div class="algorithm-pipeline-item"><div class="algorithm-pipeline-marker">' +
            esc(step.step) +
            '</div><div class="algorithm-pipeline-content"><div class="algorithm-pipeline-head"><strong>' +
            esc(step.name) +
            '</strong><span class="algorithm-pipeline-status">' +
            esc(step.status) +
            '</span></div><p class="algorithm-pipeline-text mb-0">' +
            esc(step.explanation) +
            "</p></div></div>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function renderSoftmaxTable(rows) {
    if (!rows || !rows.length) return "";
    var body = rows
      .map(function (r) {
        return (
          '<tr class="' +
          (r.status ? "algorithm-row-top" : "") +
          '"><td>' +
          esc(r.label) +
          "</td><td><code>" +
          esc(r.softmax_score != null ? Number(r.softmax_score).toFixed(4) : "—") +
          '</code></td><td><strong>' +
          esc(r.percent) +
          '%</strong></td><td>' +
          (r.status ? '<span class="badge bg-teal-soft">' + esc(r.status) + "</span>" : "—") +
          "</td></tr>"
        );
      })
      .join("");
    return (
      '<div class="table-responsive algorithm-table-wrap"><table class="table table-sm algorithm-table">' +
      "<thead><tr><th>Class</th><th>Softmax Score</th><th>Percentage</th><th>Status</th></tr></thead>" +
      "<tbody>" +
      body +
      "</tbody></table></div>"
    );
  }

  function renderRanking(items) {
    if (!items || !items.length) return "";
    return items
      .map(function (item) {
        return (
          '<div class="prob-item compact algorithm-rank-item"><div class="prob-item-header"><div class="d-flex align-items-center gap-2 flex-wrap">' +
          '<span class="rank-number">#' +
          esc(item.rank) +
          '</span><span class="class-pill ' +
          esc(item.key) +
          '">' +
          esc(item.label) +
          "</span></div><strong>" +
          esc(item.percent) +
          '%</strong></div><div class="prob-bar-track"><div class="prob-bar-fill bar-' +
          esc(item.key) +
          '" style="width:' +
          esc(item.percent) +
          '%;"></div></div></div>'
        );
      })
      .join("");
  }

  function renderFormulas(formulas) {
    if (!formulas || !formulas.length) return "";
    var blocks = formulas
      .map(function (f, idx) {
        return (
          '<div class="algorithm-formula-block"><div class="algorithm-formula-title">' +
          (idx + 1) +
          ". " +
          esc(f.title) +
          '</div><pre class="algorithm-formula-code"><code>' +
          esc(f.code) +
          "</code></pre></div>"
        );
      })
      .join("");
    return (
      '<div class="algorithm-formulas-card"><h6 class="algorithm-subsection-title"><i class="bi bi-braces me-1"></i>Rumus yang Digunakan</h6>' +
      blocks +
      "</div>"
    );
  }

  function renderAlgorithmCalculation(calc) {
    if (!calc) {
      return '<p class="text-muted small mb-0">Detail perhitungan algoritma tidak tersedia untuk record ini.</p>';
    }

    var margin = calc.margin_calculation || {};
    var conf = calc.confidence_calculation || {};
    var finalBlock = calc.final_decision_block || {};
    var formulas = calc.formulas_id || calc.formulas || [];

    return (
      '<div class="algorithm-honesty-note"><i class="bi bi-info-circle me-1"></i>' +
      esc(calc.technical_honesty_note) +
      "</div>" +
      '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">A. Ringkasan Perhitungan</h6>' +
      renderSummaryGrid(calc.summary_grid) +
      "</div>" +
      (calc.pipeline_steps && calc.pipeline_steps.length
        ? '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">B. Tahapan Algoritma</h6>' +
          renderPipeline(calc.pipeline_steps) +
          "</div>"
        : "") +
      '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">C. Softmax Probability</h6>' +
      renderSoftmaxTable(calc.softmax_table) +
      "</div>" +
      '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">D. Ranking Kelas</h6>' +
      renderRanking(calc.ranking_items || calc.softmax_table) +
      "</div>" +
      '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">E. Perhitungan Top-2 Margin</h6>' +
      '<div class="algorithm-calc-box">' +
      (margin.calculation_line
        ? '<div class="algorithm-calculation-line">' + esc(margin.calculation_line) + "</div>"
        : "") +
      '<p class="algorithm-margin-note mb-0">' +
      esc(margin.interpretation || "") +
      "</p></div></div>" +
      '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">F. Validasi Label Pembanding</h6>' +
      '<p class="algorithm-validation-note mb-0">' +
      esc(calc.validation_explanation || "") +
      "</p></div>" +
      '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">Kategori Confidence</h6>' +
      '<div class="algorithm-calc-box"><div>Confidence = <strong>' +
      esc(conf.confidence_pct) +
      '%</strong></div><div><code>' +
      esc(conf.threshold_line || "") +
      '</code></div><div class="mt-2">Confidence Category = <span class="algorithm-conf-badge conf-' +
      esc(conf.badge_tone || "info") +
      '">' +
      esc(conf.category || calc.confidence_level || "") +
      "</span></div></div></div>" +
      '<div class="algorithm-subsection"><h6 class="algorithm-subsection-title">G. Keputusan Akhir</h6>' +
      '<p class="algorithm-final-note mb-0">' +
      esc(finalBlock.explanation || "") +
      "</p></div>" +
      renderFormulas(formulas) +
      '<div class="algorithm-interpretation-card"><h6 class="algorithm-subsection-title"><i class="bi bi-mortarboard me-1"></i>H. Interpretasi Hasil</h6>' +
      '<p class="algorithm-interpretation-text mb-0">' +
      esc(calc.interpretation_id || calc.interpretation || "") +
      "</p></div>"
    );
  }

  global.AlgorithmCalculationRender = {
    render: renderAlgorithmCalculation,
  };
})(window);
