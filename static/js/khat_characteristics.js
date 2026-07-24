/**
 * Khat class visual characteristics — client-side catalog and render helpers.
 * Catalog is injected from the server as window.KHAT_CHARACTERISTICS when available.
 */
(function (global) {
  "use strict";

  var FALLBACK_CATALOG = {
    diwani: {
      class_key: "diwani",
      class_display: "Diwani",
      title: "Ciri Utama Khat Diwani",
      description:
        "Khat Diwani memiliki karakter tulisan yang lentur, elegan, dan mengalir. Bentuk hurufnya banyak menggunakan lengkungan halus, susunan huruf cenderung dekoratif, serta memiliki irama visual yang lembut.",
      characteristics: [
        "Bentuk huruf melengkung dan lentur.",
        "Komposisi tulisan mengalir dan elegan.",
        "Banyak lekukan halus.",
        "Dekoratif, tetapi tidak terlalu padat.",
        "Keterbacaan sedang karena bentuknya artistik.",
      ],
    },
    diwani_jali: {
      class_key: "diwani_jali",
      class_display: "Diwani Jali",
      title: "Ciri Utama Khat Diwani Jali",
      description:
        "Khat Diwani Jali merupakan pengembangan dari Diwani yang lebih padat, rumit, dan penuh ornamen. Gaya ini memiliki susunan huruf yang kompleks, banyak titik hias, serta ruang kosong yang lebih sedikit.",
      characteristics: [
        "Tulisan sangat dekoratif dan kompleks.",
        "Komposisi lebih padat dibanding Diwani biasa.",
        "Banyak titik, ornamen, dan hiasan visual.",
        "Ruang kosong lebih sedikit.",
        "Kesan visual mewah, resmi, dan artistik.",
      ],
    },
    naskhi: {
      class_key: "naskhi",
      class_display: "Naskhi",
      title: "Ciri Utama Khat Naskhi",
      description:
        "Khat Naskhi memiliki bentuk huruf yang rapi, jelas, dan mudah dibaca. Gaya ini sering digunakan dalam penulisan mushaf, buku, dan teks Arab panjang karena susunan hurufnya lebih teratur.",
      characteristics: [
        "Huruf rapi, sederhana, dan mudah dibaca.",
        "Bentuk tulisan cenderung horizontal dan teratur.",
        "Minim ornamen.",
        "Spasi antarhuruf dan antarbaris lebih jelas.",
        "Cocok untuk teks panjang dan penulisan formal.",
      ],
    },
    tsuluts: {
      class_key: "tsuluts",
      class_display: "Tsuluts",
      title: "Ciri Utama Khat Tsuluts",
      description:
        "Khat Tsuluts memiliki karakter megah, tegas, dan monumental. Hurufnya cenderung besar, memiliki tarikan garis panjang, lengkungan luas, serta sering digunakan pada dekorasi masjid, judul, dan karya kaligrafi besar.",
      characteristics: [
        "Huruf besar, tinggi, dan dominan.",
        "Tarikan garis panjang dan tegas.",
        "Lengkungan besar dan artistik.",
        "Komposisi terlihat megah dan monumental.",
        "Dekoratif, tetapi tidak sepadat Diwani Jali.",
      ],
    },
  };

  var DISCLAIMER =
    "Catatan: Ciri utama ini digunakan sebagai informasi pendukung. Keputusan akhir tetap mengacu pada hasil prediksi model, confidence score, validation status, dan manual review jika diperlukan.";

  var MISMATCH_WARNING =
    "Perhatian: Ciri utama yang ditampilkan mengikuti kelas hasil prediksi model. Karena hasil prediksi tidak sesuai dengan expected class, pengguna disarankan membandingkan ciri ini dengan label gambar secara manual.";

  function catalog() {
    return global.KHAT_CHARACTERISTICS || FALLBACK_CATALOG;
  }

  function normalizeKey(classKey) {
    if (!classKey) return null;
    return String(classKey).trim().toLowerCase().replace(/-/g, "_").replace(/ /g, "_");
  }

  function getCharacteristics(classKey) {
    var key = normalizeKey(classKey);
    return key ? catalog()[key] || null : null;
  }

  function listItems(items) {
    return (items || [])
      .map(function (item) {
        return (
          '<li><i class="bi bi-check2-circle"></i><span>' +
          item +
          "</span></li>"
        );
      })
      .join("");
  }

  function renderCharacteristicsBlock(chars, label) {
    if (!chars) return "";
    var bullets = chars.short_characteristics || chars.characteristics || [];
    var desc = chars.short_description || chars.description || "";
    return (
      '<div class="khat-chars-compare-col">' +
      '<div class="khat-chars-compare-head">' +
      '<span class="class-pill ' +
      chars.class_key +
      '">' +
      chars.class_display +
      "</span>" +
      "<small>" +
      label +
      "</small>" +
      "</div>" +
      '<p class="khat-chars-compare-desc">' +
      desc +
      "</p>" +
      '<ul class="khat-chars-list compact">' +
      listItems(bullets) +
      "</ul>" +
      "</div>"
    );
  }

  function renderHistoryCharacteristics(payload) {
    if (!payload || !payload.predicted) {
      return '<p class="hist-detail-chars-empty">Ciri utama tidak tersedia untuk prediksi ini.</p>';
    }

    var html = '<div class="hist-detail-chars-inner">';

    if (payload.mismatch_warning) {
      html +=
        '<div class="hist-detail-chars-alert">' +
        '<i class="bi bi-exclamation-triangle-fill" aria-hidden="true"></i>' +
        "<div>" +
        payload.mismatch_warning;
      if (payload.expected_display) {
        html +=
          '<p class="mb-0 mt-1">Expected Class: <strong>' +
          payload.expected_display +
          "</strong> · Predicted Class: <strong>" +
          payload.predicted_display +
          "</strong></p>";
      }
      html += "</div></div>";
    }

    html +=
      '<div class="hist-detail-chars-grid">' +
      '<div class="hist-detail-chars-desc">' +
      "<h6>" +
      payload.predicted.title +
      "</h6>" +
      '<p class="hist-detail-chars-text">' +
      payload.predicted.description +
      "</p>" +
      "</div>" +
      '<div class="hist-detail-chars-traits">' +
      '<p class="hist-detail-traits-label">Karakteristik utama</p>' +
      '<ul class="hist-detail-trait-list">' +
      listItems(payload.predicted.characteristics) +
      "</ul>" +
      "</div>" +
      "</div>";

    if (payload.show_comparison && payload.expected) {
      var predShort = payload.predicted_comparison || payload.predicted;
      var expShort = payload.expected_comparison || payload.expected;
      html +=
        '<div class="hist-detail-chars-compare">' +
        '<h6 class="hist-detail-compare-title"><i class="bi bi-arrow-left-right me-1"></i>Perbandingan dengan Expected Class</h6>' +
        '<div class="hist-detail-compare-grid">' +
        renderCharacteristicsBlock(predShort, "Predicted Class") +
        renderCharacteristicsBlock(expShort, "Expected Class") +
        "</div></div>";
    }

    html +=
      '<p class="hist-detail-chars-disclaimer">' +
      '<i class="bi bi-info-circle" aria-hidden="true"></i>' +
      (payload.disclaimer_note || DISCLAIMER) +
      "</p></div>";

    return html;
  }

  global.KhatCharacteristics = {
    catalog: catalog,
    get: getCharacteristics,
    renderHistoryCharacteristics: renderHistoryCharacteristics,
    DISCLAIMER: DISCLAIMER,
    MISMATCH_WARNING: MISMATCH_WARNING,
  };
})(window);
