/**
 * Login page — password toggle, remember me, forgot password hint, validation & loading UX.
 */
(function () {
  "use strict";

  const REMEMBER_KEY = "khat_login_remember_username";
  const form = document.getElementById("loginForm");
  const usernameInput = document.getElementById("username");
  const passwordInput = document.getElementById("password");
  const rememberCheckbox = document.getElementById("rememberMe");
  const forgotBtn = document.getElementById("forgotPasswordBtn");
  const forgotNote = document.getElementById("forgotPasswordNote");
  const toggleBtn = document.getElementById("passwordToggle");
  const submitBtn = document.getElementById("loginSubmitBtn");

  function setFieldError(input, errorEl, message) {
    if (!input) return;
    const wrap = input.closest(".input-wrap");
    if (message) {
      input.classList.add("is-invalid");
      if (wrap) wrap.classList.add("is-invalid");
      if (errorEl) errorEl.textContent = message;
    } else {
      input.classList.remove("is-invalid");
      if (wrap) wrap.classList.remove("is-invalid");
      if (errorEl) errorEl.textContent = "";
    }
  }

  function validateField(input, errorEl, emptyMessage) {
    if (!input) return true;
    if (!input.value.trim()) {
      setFieldError(input, errorEl, emptyMessage);
      return false;
    }
    setFieldError(input, errorEl, "");
    return true;
  }

  function restoreRememberedUsername() {
    if (!usernameInput || !rememberCheckbox) return;
    try {
      const saved = localStorage.getItem(REMEMBER_KEY);
      if (saved) {
        usernameInput.value = saved;
        rememberCheckbox.checked = true;
      }
    } catch (_err) {
      /* localStorage unavailable */
    }
  }

  function persistRememberedUsername() {
    if (!usernameInput || !rememberCheckbox) return;
    try {
      if (rememberCheckbox.checked && usernameInput.value.trim()) {
        localStorage.setItem(REMEMBER_KEY, usernameInput.value.trim());
      } else {
        localStorage.removeItem(REMEMBER_KEY);
      }
    } catch (_err) {
      /* localStorage unavailable */
    }
  }

  restoreRememberedUsername();

  if (forgotBtn && forgotNote) {
    forgotBtn.addEventListener("click", function () {
      forgotNote.classList.toggle("d-none");
      if (!forgotNote.classList.contains("d-none")) {
        forgotNote.setAttribute("tabindex", "-1");
        forgotNote.focus();
      }
    });
  }

  if (toggleBtn && passwordInput) {
    toggleBtn.addEventListener("click", function () {
      const isHidden = passwordInput.type === "password";
      passwordInput.type = isHidden ? "text" : "password";
      const icon = toggleBtn.querySelector("i");
      if (icon) {
        icon.className = isHidden ? "bi bi-eye-slash" : "bi bi-eye";
      }
      toggleBtn.setAttribute("aria-label", isHidden ? "Sembunyikan password" : "Tampilkan password");
      toggleBtn.setAttribute("aria-pressed", isHidden ? "true" : "false");
    });

    toggleBtn.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleBtn.click();
      }
    });
  }

  [usernameInput, passwordInput].forEach(function (input) {
    if (!input) return;
    input.addEventListener("input", function () {
      const errorEl = document.getElementById(input.id + "Error");
      if (input.classList.contains("is-invalid")) {
        setFieldError(input, errorEl, "");
      }
    });
    input.addEventListener("blur", function () {
      const errorEl = document.getElementById(input.id + "Error");
      if (!input.value.trim()) {
        setFieldError(input, errorEl, "");
        return;
      }
      setFieldError(input, errorEl, "");
    });
  });

  if (form && submitBtn) {
    form.addEventListener("submit", function (e) {
      const usernameOk = validateField(
        usernameInput,
        document.getElementById("usernameError"),
        "Username wajib diisi."
      );
      const passwordOk = validateField(
        passwordInput,
        document.getElementById("passwordError"),
        "Password wajib diisi."
      );

      if (!usernameOk || !passwordOk) {
        e.preventDefault();
        const firstInvalid = form.querySelector(".is-invalid");
        if (firstInvalid) firstInvalid.focus();
        return;
      }

      persistRememberedUsername();

      submitBtn.disabled = true;
      submitBtn.classList.add("is-loading");
      const spinner = submitBtn.querySelector(".login-btn-spinner");
      const icon = submitBtn.querySelector(".login-btn-icon");
      const btnText = submitBtn.querySelector(".login-btn-text");
      const loadingText = submitBtn.querySelector(".login-btn-loading-text");
      if (icon) icon.classList.add("d-none");
      if (btnText) btnText.classList.add("d-none");
      if (loadingText) {
        loadingText.classList.remove("d-none");
      } else if (spinner) {
        spinner.classList.remove("d-none");
      }
    });
  }
})();
