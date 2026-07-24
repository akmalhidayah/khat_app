/**
 * Browser-side Teachable Machine model loader and predictor.
 * Uses @tensorflow/tfjs and @teachablemachine/image from CDN.
 */
(function (global) {
  "use strict";

  const DEFAULT_MODEL_URL = "/static/model/model.json";
  const DEFAULT_METADATA_URL = "/static/model/metadata.json";

  let model = null;
  let loadPromise = null;
  let modelURL = DEFAULT_MODEL_URL;
  let metadataURL = DEFAULT_METADATA_URL;

  const MIN_CONFIDENCE = 0.70;
  const MIN_MARGIN = 0.15;
  const CANONICAL_CLASSES = ["naskhi", "diwani", "diwani_jali", "tsuluts"];

  function applyRecognitionGate(sorted) {
    if (!sorted || !sorted.length) {
      return { rejected: true, reason: "Tidak ada prediksi." };
    }
    const top = sorted[0];
    const second = sorted[1] || null;
    const confPct = top.percent;
    const marginPct = second ? top.percent - second.percent : confPct;
    const reasons = [];
    if (!top.classKey || CANONICAL_CLASSES.indexOf(top.classKey) === -1) {
      reasons.push("Gaya khat tidak termasuk kelas dataset training.");
    }
    if (confPct < MIN_CONFIDENCE * 100) {
      reasons.push("Tingkat keyakinan model terlalu rendah.");
    }
    if (marginPct < MIN_MARGIN * 100) {
      reasons.push("Hasil prediksi kurang jelas antar kelas.");
    }
    if (reasons.length) {
      return { rejected: true, reason: "Khat tidak dikenali. " + reasons.join(" ") };
    }
    return { rejected: false };
  }

  function normalizeLabel(raw) {
    const text = String(raw || "").trim().toLowerCase().replace(/-/g, "_");
    const compact = text.replace(/\s+/g, "_");
    if (compact.includes("diwani") && compact.includes("jali")) return "diwani_jali";
    if (compact.includes("naskhi") || compact.includes("naski")) return "naskhi";
    if (compact.includes("diwani")) return "diwani";
    if (/(tsuluts|tsuluth|thuluth|sulus)/.test(compact)) return "tsuluts";
    return compact;
  }

  function ensureLibraries() {
    if (!global.tmImage) {
      throw new Error("Teachable Machine image library is not loaded.");
    }
  }

  async function load(options) {
    if (model) return model;
    if (loadPromise) return loadPromise;

    loadPromise = (async function () {
      ensureLibraries();
      modelURL = (options && options.modelURL) || DEFAULT_MODEL_URL;
      metadataURL = (options && options.metadataURL) || DEFAULT_METADATA_URL;

      model = await global.tmImage.load(modelURL, metadataURL);

      console.log("Teachable Machine model loaded successfully");
      console.log("Model URL:", modelURL);
      console.log("Metadata URL:", metadataURL);
      if (typeof model.getClassLabels === "function") {
        console.log("Class Labels:", model.getClassLabels());
      } else if (model._metadata && model._metadata.labels) {
        console.log("Class Labels:", model._metadata.labels);
      }

      return model;
    })().catch(function (error) {
      loadPromise = null;
      console.error("Teachable Machine model failed to load:", error);
      throw error;
    });

    return loadPromise;
  }

  async function predictImageElement(imageElement) {
    const activeModel = await load();
    const predictions = await activeModel.predict(imageElement, false);
    const sorted = predictions
      .slice()
      .sort(function (a, b) {
        return b.probability - a.probability;
      })
      .map(function (item, index) {
        return {
          rank: index + 1,
          className: item.className,
          classKey: normalizeLabel(item.className),
          probability: item.probability,
          percent: Math.round(item.probability * 10000) / 100,
        };
      });

    const top = sorted[0] || null;
    const gate = applyRecognitionGate(sorted);
    if (gate.rejected) {
      return {
        predicted_class: null,
        predicted_label: null,
        confidence: 0,
        confidence_pct: 0,
        sorted_predictions: sorted,
        input_status: "unrecognized",
        rejection_reason: gate.reason,
        message: "Khat tidak dikenali.",
      };
    }
    return {
      predicted_class: top ? top.classKey : null,
      predicted_label: top ? top.className : null,
      confidence: top ? top.probability : 0,
      confidence_pct: top ? top.percent : 0,
      sorted_predictions: sorted,
    };
  }

  async function verifyOnPage(imageSelector) {
    const img =
      typeof imageSelector === "string"
        ? document.querySelector(imageSelector)
        : imageSelector;
    if (!img || !img.complete || !img.naturalWidth) {
      return null;
    }
    try {
      const result = await predictImageElement(img);
      console.log("Teachable Machine browser prediction:", result);
      return result;
    } catch (error) {
      console.warn("Teachable Machine browser verification skipped:", error.message);
      return null;
    }
  }

  global.KhatTeachableMachine = {
    load: load,
    predictImageElement: predictImageElement,
    verifyOnPage: verifyOnPage,
    isLoaded: function () {
      return Boolean(model);
    },
    getModelURLs: function () {
      return { modelURL: modelURL, metadataURL: metadataURL };
    },
  };
})(window);
