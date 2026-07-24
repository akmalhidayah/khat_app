/**
 * Classify Image page — upload, preview, validation, submit state.
 */
(function () {
  "use strict";

  const MAX_BYTES = 10 * 1024 * 1024;
  const ALLOWED = ["jpg", "jpeg", "png", "webp"];

  const input = document.getElementById("classifyImageInput");
  const dropzone = document.getElementById("uploadDropzone");
  const previewWrap = document.getElementById("previewWrap");
  const previewImage = document.getElementById("previewImage");
  const previewEmpty = document.getElementById("previewEmpty");
  const invalidNote = document.getElementById("invalidTypeNote");
  const fileMeta = document.getElementById("fileMeta");
  const fileNameEl = document.getElementById("fileName");
  const fileTypeEl = document.getElementById("fileType");
  const fileSizeEl = document.getElementById("fileSize");
  const fileDimsEl = document.getElementById("fileDimensions");
  const submitBtn = document.getElementById("classifySubmitBtn");
  const clearBtn = document.getElementById("clearFileBtn");
  const zoomBtn = document.getElementById("previewZoomBtn");
  const form = document.getElementById("classifyForm");
  const browseBtn = document.getElementById("browseImageBtn");
  const cameraBtn = document.getElementById("cameraCaptureBtn");
  const cameraModalEl = document.getElementById("classifyCameraModal");
  const cameraVideo = document.getElementById("classifyCameraVideo");
  const cameraCanvas = document.getElementById("classifyCameraCanvas");
  const cameraSnapBtn = document.getElementById("classifyCameraSnapBtn");
  const cameraSwitchBtn = document.getElementById("classifyCameraSwitchBtn");
  const cameraErrorEl = document.getElementById("classifyCameraError");
  const cameraPlaceholder = document.getElementById("classifyCameraPlaceholder");
  const cameraPlaceholderText = document.getElementById("classifyCameraPlaceholderText");
  const cameraStage = document.getElementById("classifyCameraStage");
  const cameraRetryBtn = document.getElementById("classifyCameraRetryBtn");
  const cameraFallbackWrap = document.getElementById("classifyCameraFallbackWrap");
  const cameraFallbackBtn = document.getElementById("classifyCameraFallbackBtn");
  const cameraFallbackInput = document.getElementById("classifyCameraFallbackInput");

  if (!input || !dropzone || !previewWrap || !previewImage || !submitBtn || !form) {
    return;
  }

  let submitting = false;
  let cameraStream = null;
  let cameraFacingMode = "user";
  let cameraStarting = false;
  let cameraStartToken = 0;
  let cameraDevices = [];
  let cameraDeviceIndex = 0;

  const toSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  };

  const setBtnIdle = () => {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="bi bi-image me-2"></i><span>Pilih Gambar Dulu</span>';
    if (clearBtn) clearBtn.classList.add("d-none");
  };

  const setBtnReady = () => {
    submitBtn.disabled = false;
    submitBtn.innerHTML =
      '<i class="bi bi-shield-check me-2"></i><span>Validasi &amp; Klasifikasi</span>';
    if (clearBtn) clearBtn.classList.remove("d-none");
  };

  const setBtnLoading = () => {
    submitBtn.disabled = true;
    submitBtn.innerHTML =
      '<span class="spinner-border spinner-border-sm me-2" role="status"></span><span>Menganalisis...</span>';
  };

  const showError = (message) => {
    if (!invalidNote) return;
    invalidNote.textContent = message;
    invalidNote.classList.remove("d-none");
  };

  const hideError = () => {
    if (invalidNote) invalidNote.classList.add("d-none");
  };

  const resetView = () => {
    previewWrap.classList.add("d-none");
    previewEmpty.classList.remove("d-none");
    hideError();
    if (fileMeta) fileMeta.classList.add("d-none");
    if (zoomBtn) zoomBtn.classList.add("d-none");
    previewImage.removeAttribute("src");
    if (fileDimsEl) fileDimsEl.textContent = "—";
    setBtnIdle();
    submitting = false;
  };

  const validateFile = (file) => {
    if (!file) return { ok: false, message: "Tidak ada file dipilih." };
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!ALLOWED.includes(ext)) {
      return {
        ok: false,
        message: "Format tidak valid. Gunakan JPG, JPEG, PNG, atau WEBP.",
      };
    }
    if (file.size > MAX_BYTES) {
      return { ok: false, message: "File terlalu besar. Maksimum 10 MB." };
    }
    return { ok: true, ext };
  };

  const assignFileToInput = (file) => {
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    applyFile(file);
  };

  const applyFile = (file) => {
    const check = validateFile(file);
    if (!check.ok) {
      showError(check.message);
      previewWrap.classList.add("d-none");
      previewEmpty.classList.remove("d-none");
      if (fileMeta) fileMeta.classList.add("d-none");
      if (zoomBtn) zoomBtn.classList.add("d-none");
      setBtnIdle();
      return;
    }

    hideError();
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.onload = () => {
        if (fileDimsEl) {
          fileDimsEl.textContent =
            previewImage.naturalWidth + " × " + previewImage.naturalHeight + " px";
        }
        if (window.KhatTeachableMachine && document.querySelector(".classify-v2[data-tm-active='1']")) {
          window.KhatTeachableMachine.load()
            .then(function () {
              return window.KhatTeachableMachine.predictImageElement(previewImage);
            })
            .then(function (result) {
              console.log("Teachable Machine preview prediction:", result);
            })
            .catch(function () {
              /* TM files may not be installed yet */
            });
        }
      };
      previewImage.src = e.target.result;
      previewWrap.classList.remove("d-none");
      previewEmpty.classList.add("d-none");
      if (zoomBtn) zoomBtn.classList.remove("d-none");
      if (fileMeta) fileMeta.classList.remove("d-none");
    };
    reader.onerror = () => {
      showError("Gagal membaca file. Coba gambar lain.");
      resetView();
    };
    reader.readAsDataURL(file);

    if (fileNameEl) fileNameEl.textContent = file.name;
    if (fileTypeEl) fileTypeEl.textContent = (file.type || check.ext).toUpperCase();
    if (fileSizeEl) fileSizeEl.textContent = toSize(file.size);
    setBtnReady();
  };

  input.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) {
      resetView();
      return;
    }
    applyFile(file);
  });

  if (browseBtn) {
    browseBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      input.click();
    });
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#browseImageBtn, #cameraCaptureBtn")) return;
    input.click();
  });

  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    assignFileToInput(file);
  });

  const showCameraError = (message) => {
    if (!cameraErrorEl) return;
    cameraErrorEl.textContent = message;
    cameraErrorEl.classList.remove("d-none");
  };

  const hideCameraError = () => {
    if (cameraErrorEl) cameraErrorEl.classList.add("d-none");
  };

  const setCameraPlaceholder = (visible, message) => {
    if (cameraPlaceholder) {
      cameraPlaceholder.classList.toggle("d-none", !visible);
    }
    if (cameraVideo) {
      cameraVideo.classList.toggle("d-none", visible);
    }
    if (cameraStage) {
      cameraStage.classList.toggle("is-loading", visible);
    }
    if (cameraPlaceholderText && message) {
      cameraPlaceholderText.textContent = message;
    }
  };

  const setCameraFallbackVisible = (visible) => {
    if (cameraFallbackWrap) {
      cameraFallbackWrap.classList.toggle("d-none", !visible);
    }
  };

  const setCameraRetryVisible = (visible) => {
    if (cameraRetryBtn) {
      cameraRetryBtn.classList.toggle("d-none", !visible);
    }
  };

  const getUserMediaFn = () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      return (constraints) => navigator.mediaDevices.getUserMedia(constraints);
    }
    const legacy =
      navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!legacy) return null;
    return (constraints) =>
      new Promise((resolve, reject) => {
        legacy.call(navigator, constraints, resolve, reject);
      });
  };

  const refreshCameraDevices = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      cameraDevices = devices.filter((device) => device.kind === "videoinput" && device.deviceId);
      if (cameraDeviceIndex >= cameraDevices.length) {
        cameraDeviceIndex = 0;
      }
    } catch (_err) {
      cameraDevices = [];
    }
  };

  const buildCameraAttempts = () => {
    const gum = getUserMediaFn();
    if (!gum) return [];

    const attempts = [
      { video: true, audio: false },
      { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: cameraFacingMode }, audio: false },
      { video: { facingMode: { ideal: cameraFacingMode } }, audio: false },
    ];

    if (cameraDevices.length > 0) {
      const device = cameraDevices[cameraDeviceIndex];
      if (device && device.deviceId) {
        attempts.unshift({ video: { deviceId: { ideal: device.deviceId } }, audio: false });
        attempts.unshift({ video: { deviceId: device.deviceId }, audio: false });
      }
    }

    return attempts.map((constraints) => () => gum(constraints));
  };

  const describeCameraError = (err) => {
    if (!err) {
      return "Gagal membuka kamera. Pastikan kamera terhubung dan tidak dipakai aplikasi lain.";
    }
    if (err.message === "unsupported") {
      return "Browser tidak mendukung akses kamera. Gunakan Chrome/Edge terbaru atau unggah file.";
    }
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      return "Izin kamera ditolak. Klik ikon kamera/gembok di address bar, izinkan akses, lalu coba lagi.";
    }
    if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      return "Kamera tidak ditemukan. Pastikan webcam terhubung dan driver sudah terpasang.";
    }
    if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      return "Kamera sedang dipakai aplikasi lain (Zoom, Teams, dll). Tutup aplikasi tersebut lalu klik Coba Lagi.";
    }
    if (err.name === "OverconstrainedError") {
      return "Pengaturan kamera tidak didukung perangkat ini. Klik Coba Lagi atau gunakan Buka Kamera Sistem.";
    }
    if (err.name === "SecurityError") {
      return "Akses kamera diblokir. Buka situs via http://127.0.0.1 (bukan IP lain) dan izinkan kamera.";
    }
    return "Gagal membuka kamera (" + (err.name || "error") + "). Klik Coba Lagi.";
  };

  const requestCameraStream = async () => {
    const gum = getUserMediaFn();
    if (!gum) {
      throw Object.assign(new Error("unsupported"), { name: "NotSupportedError" });
    }

    await refreshCameraDevices();

    let lastError = null;
    for (const attempt of buildCameraAttempts()) {
      try {
        const stream = await attempt();
        await refreshCameraDevices();
        if (cameraSwitchBtn) {
          cameraSwitchBtn.classList.toggle("d-none", cameraDevices.length < 2);
        }
        return stream;
      } catch (err) {
        lastError = err;
        if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
          throw err;
        }
      }
    }

    if (
      lastError &&
      (lastError.name === "NotReadableError" || lastError.name === "TrackStartError")
    ) {
      await new Promise((resolve) => window.setTimeout(resolve, 800));
      try {
        const stream = await gum({ video: true, audio: false });
        await refreshCameraDevices();
        return stream;
      } catch (retryErr) {
        lastError = retryErr;
      }
    }

    throw lastError || new Error("camera-unavailable");
  };

  const bindCameraVideo = async () => {
    if (!cameraVideo || !cameraStream) return;
    cameraVideo.srcObject = cameraStream;
    cameraVideo.setAttribute("playsinline", "");
    cameraVideo.muted = true;
    cameraVideo.autoplay = true;

    await new Promise((resolve, reject) => {
      const timeoutId = window.setTimeout(() => {
        reject(new Error("video-timeout"));
      }, 12000);

      const finish = (fn) => {
        window.clearTimeout(timeoutId);
        fn();
      };

      const startPlayback = () => {
        cameraVideo
          .play()
          .then(() => finish(resolve))
          .catch((playErr) => finish(() => reject(playErr)));
      };

      if (cameraVideo.readyState >= 1) {
        startPlayback();
        return;
      }
      cameraVideo.onloadedmetadata = startPlayback;
    });
  };

  const releaseCameraStream = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }
    if (cameraVideo) {
      cameraVideo.srcObject = null;
    }
    if (cameraSnapBtn) {
      cameraSnapBtn.disabled = true;
    }
  };

  const stopCamera = () => {
    cameraStartToken += 1;
    cameraStarting = false;
    releaseCameraStream();
  };

  const startCamera = async () => {
    if (cameraStarting) return;

    cameraStarting = true;
    const token = ++cameraStartToken;
    releaseCameraStream();
    hideCameraError();
    setCameraFallbackVisible(false);
    setCameraRetryVisible(false);
    setCameraPlaceholder(true, "Menghidupkan kamera...");
    if (cameraSnapBtn) {
      cameraSnapBtn.disabled = true;
    }

    try {
      const stream = await requestCameraStream();
      if (token !== cameraStartToken) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      cameraStream = stream;
      setCameraPlaceholder(false);
      await bindCameraVideo();
      if (token !== cameraStartToken) return;
      if (cameraSnapBtn) cameraSnapBtn.disabled = false;
      if (cameraSwitchBtn && cameraDevices.length > 1) {
        cameraSwitchBtn.classList.remove("d-none");
      }
    } catch (err) {
      if (token !== cameraStartToken) return;
      setCameraPlaceholder(true, "Tidak dapat mengakses kamera.");
      showCameraError(describeCameraError(err));
      setCameraRetryVisible(true);
      setCameraFallbackVisible(true);
    } finally {
      if (token === cameraStartToken) {
        cameraStarting = false;
      }
    }
  };

  const captureCameraPhoto = () => {
    if (!cameraVideo || !cameraCanvas) return;
    const vw = cameraVideo.videoWidth;
    const vh = cameraVideo.videoHeight;
    if (!vw || !vh) {
      showCameraError("Kamera belum siap. Tunggu sebentar lalu coba lagi.");
      return;
    }

    const maxDim = 1920;
    let w = vw;
    let h = vh;
    if (Math.max(w, h) > maxDim) {
      const scale = maxDim / Math.max(w, h);
      w = Math.round(w * scale);
      h = Math.round(h * scale);
    }

    cameraCanvas.width = w;
    cameraCanvas.height = h;
    const ctx = cameraCanvas.getContext("2d");
    if (!ctx) {
      showCameraError("Gagal menyiapkan canvas untuk capture.");
      return;
    }
    ctx.drawImage(cameraVideo, 0, 0, w, h);

    cameraCanvas.toBlob(
      (blob) => {
        if (!blob) {
          showCameraError("Gagal mengambil foto. Coba lagi.");
          return;
        }
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        const file = new File([blob], "camera-capture-" + stamp + ".jpg", {
          type: "image/jpeg",
          lastModified: Date.now(),
        });
        assignFileToInput(file);
        if (cameraModalEl && window.bootstrap) {
          bootstrap.Modal.getOrCreateInstance(cameraModalEl).hide();
        }
      },
      "image/jpeg",
      0.92
    );
  };

  const openCameraModal = () => {
    if (!window.bootstrap || !cameraModalEl) {
      showError("Bootstrap belum dimuat. Muat ulang halaman lalu coba lagi.");
      return;
    }
    bootstrap.Modal.getOrCreateInstance(cameraModalEl).show();
    startCamera();
  };

  const initCameraFeature = () => {
    if (!cameraBtn || !cameraModalEl) return;

    cameraBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      openCameraModal();
    });

    cameraModalEl.addEventListener("hidden.bs.modal", () => {
      stopCamera();
      hideCameraError();
      setCameraPlaceholder(false);
      setCameraFallbackVisible(false);
      setCameraRetryVisible(false);
      if (cameraStage) cameraStage.classList.remove("is-loading");
    });
  };

  initCameraFeature();

  if (cameraRetryBtn) {
    cameraRetryBtn.addEventListener("click", () => {
      startCamera();
    });
  }

  if (cameraFallbackBtn && cameraFallbackInput) {
    cameraFallbackBtn.addEventListener("click", () => {
      cameraFallbackInput.value = "";
      cameraFallbackInput.click();
    });

    cameraFallbackInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      assignFileToInput(file);
      if (cameraModalEl && window.bootstrap) {
        bootstrap.Modal.getOrCreateInstance(cameraModalEl).hide();
      }
    });
  }

  if (cameraSnapBtn) {
    cameraSnapBtn.addEventListener("click", captureCameraPhoto);
  }

  if (cameraSwitchBtn) {
    cameraSwitchBtn.addEventListener("click", () => {
      if (cameraDevices.length > 1) {
        cameraDeviceIndex = (cameraDeviceIndex + 1) % cameraDevices.length;
      } else {
        cameraFacingMode = cameraFacingMode === "environment" ? "user" : "environment";
      }
      startCamera();
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      input.value = "";
      resetView();
    });
  }

  if (zoomBtn) {
    zoomBtn.addEventListener("click", () => {
      const src = previewImage.getAttribute("src");
      if (!src) return;
      const modalImg = document.getElementById("classifyPreviewModalImg");
      const modalEl = document.getElementById("classifyPreviewModal");
      if (modalImg && modalEl && window.bootstrap) {
        modalImg.src = src;
        bootstrap.Modal.getOrCreateInstance(modalEl).show();
      }
    });
  }

  form.addEventListener("submit", async (e) => {
    if (submitting) {
      e.preventDefault();
      return;
    }
    const file = input.files && input.files[0];
    const check = validateFile(file);
    if (!check.ok) {
      e.preventDefault();
      showError(check.message);
      setBtnIdle();
      return;
    }

    const tmActive = document.querySelector(".classify-v2[data-tm-active='1']");
    if (tmActive && window.KhatTeachableMachine && previewImage.getAttribute("src")) {
      e.preventDefault();
      submitting = true;
      setBtnLoading();
      try {
        const result = await window.KhatTeachableMachine.predictImageElement(previewImage);
        console.log("Teachable Machine classification prediction:", result);
        submitting = true;
        form.submit();
      } catch (err) {
        console.warn("Teachable Machine client prediction failed, submitting to server:", err);
        submitting = true;
        form.submit();
      }
      return;
    }

    submitting = true;
    setBtnLoading();
  });

  const dismissDemo = document.getElementById("dismissDemoWarning");
  const demoCard = document.getElementById("demoModelCard");
  if (dismissDemo && demoCard) {
    if (localStorage.getItem("khat_dismiss_demo_warning") === "1") {
      demoCard.classList.add("d-none");
    }
    dismissDemo.addEventListener("click", () => {
      demoCard.classList.add("d-none");
      localStorage.setItem("khat_dismiss_demo_warning", "1");
    });
  }

  resetView();

  if (document.querySelector(".classify-v2[data-tm-active='1']") && window.KhatTeachableMachine) {
    window.KhatTeachableMachine.load().catch(function () {
      /* model load will retry on submit */
    });
  }
})();
