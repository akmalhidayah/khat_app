/**
 * Register page — password toggles, client validation & loading UX.
 */
(function () {
  "use strict";

  const form = document.getElementById("registerForm");
  const nameInput = document.getElementById("name");
  const usernameInput = document.getElementById("username");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const passwordConfirmInput = document.getElementById("password_confirm");
  const submitBtn = document.getElementById("registerSubmitBtn");
  const USERNAME_RE = /^[a-zA-Z0-9_]{3,50}$/;
  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

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

  function wirePasswordToggle(btnId, input) {
    const toggleBtn = document.getElementById(btnId);
    if (!toggleBtn || !input) return;
    toggleBtn.addEventListener("click", function () {
      const isHidden = input.type === "password";
      input.type = isHidden ? "text" : "password";
      const icon = toggleBtn.querySelector("i");
      if (icon) {
        icon.className = isHidden ? "bi bi-eye-slash" : "bi bi-eye";
      }
      toggleBtn.setAttribute("aria-label", isHidden ? "Sembunyikan password" : "Tampilkan password");
      toggleBtn.setAttribute("aria-pressed", isHidden ? "true" : "false");
    });
  }

  wirePasswordToggle("passwordToggle", passwordInput);
  wirePasswordToggle("passwordConfirmToggle", passwordConfirmInput);

  [nameInput, usernameInput, emailInput, passwordInput, passwordConfirmInput].forEach(function (input) {
    if (!input) return;
    input.addEventListener("input", function () {
      const errorEl = document.getElementById(input.id + "Error");
      if (input.classList.contains("is-invalid")) {
        setFieldError(input, errorEl, "");
      }
    });
  });

  if (form && submitBtn) {
    form.addEventListener("submit", function (e) {
      let ok = true;

      if (!nameInput || nameInput.value.trim().length < 2) {
        setFieldError(nameInput, document.getElementById("nameError"), "Nama lengkap minimal 2 karakter.");
        ok = false;
      }

      const username = usernameInput ? usernameInput.value.trim() : "";
      if (!username) {
        setFieldError(usernameInput, document.getElementById("usernameError"), "Username wajib diisi.");
        ok = false;
      } else if (!USERNAME_RE.test(username)) {
        setFieldError(
          usernameInput,
          document.getElementById("usernameError"),
          "Username 3–50 karakter (huruf, angka, underscore)."
        );
        ok = false;
      }

      const email = emailInput ? emailInput.value.trim() : "";
      if (email && !EMAIL_RE.test(email)) {
        setFieldError(emailInput, document.getElementById("emailError"), "Format email tidak valid.");
        ok = false;
      }

      const password = passwordInput ? passwordInput.value.trim() : "";
      if (password.length < 6) {
        setFieldError(passwordInput, document.getElementById("passwordError"), "Password minimal 6 karakter.");
        ok = false;
      }

      const confirm = passwordConfirmInput ? passwordConfirmInput.value.trim() : "";
      if (!confirm) {
        setFieldError(
          passwordConfirmInput,
          document.getElementById("password_confirmError"),
          "Konfirmasi password wajib diisi."
        );
        ok = false;
      } else if (password !== confirm) {
        setFieldError(
          passwordConfirmInput,
          document.getElementById("password_confirmError"),
          "Konfirmasi password tidak cocok."
        );
        ok = false;
      }

      if (!ok) {
        e.preventDefault();
        const firstInvalid = form.querySelector(".is-invalid");
        if (firstInvalid) firstInvalid.focus();
        return;
      }

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
