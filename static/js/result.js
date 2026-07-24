document.addEventListener("DOMContentLoaded", function () {
    const bars = document.querySelectorAll(".prob-bar-fill[data-width]");
    bars.forEach(function (bar, index) {
        const target = bar.getAttribute("data-width");
        bar.style.width = "0%";
        setTimeout(function () {
            bar.style.width = target + "%";
        }, 80 + index * 60);
    });

    const ringWrap = document.querySelector(".confidence-ring-wrap[data-confidence]");
    if (ringWrap) {
        const value = parseFloat(ringWrap.getAttribute("data-confidence")) || 0;
        ringWrap.style.setProperty("--ring-deg", value * 3.6 + "deg");
    }

    const algoCollapse = document.getElementById("algorithmCalcCollapse");
    const algoToggle = document.getElementById("algorithmCalcToggleBtn");
    const algoSection = document.getElementById("algorithmCalcSection");
    if (algoCollapse && algoToggle && algoSection) {
        const storageKey = "algorithmCalcOpen:" + (algoSection.getAttribute("data-result-id") || "default");
        const savedOpen = sessionStorage.getItem(storageKey) === "1";
        if (savedOpen) {
            algoCollapse.classList.add("show");
            algoToggle.setAttribute("aria-expanded", "true");
            const label = algoToggle.querySelector(".toggle-label");
            if (label) label.textContent = "Sembunyikan";
            const chevron = algoToggle.querySelector(".algorithm-calc-chevron");
            if (chevron) chevron.classList.replace("bi-chevron-down", "bi-chevron-up");
        }
        algoCollapse.addEventListener("shown.bs.collapse", function () {
            sessionStorage.setItem(storageKey, "1");
            const label = algoToggle.querySelector(".toggle-label");
            if (label) label.textContent = "Sembunyikan";
            const chevron = algoToggle.querySelector(".algorithm-calc-chevron");
            if (chevron) chevron.classList.replace("bi-chevron-down", "bi-chevron-up");
        });
        algoCollapse.addEventListener("hidden.bs.collapse", function () {
            sessionStorage.setItem(storageKey, "0");
            const label = algoToggle.querySelector(".toggle-label");
            if (label) label.textContent = "Lihat Detail";
            const chevron = algoToggle.querySelector(".algorithm-calc-chevron");
            if (chevron) chevron.classList.replace("bi-chevron-up", "bi-chevron-down");
        });
    }

    if (window.KhatTeachableMachine && document.querySelector(".result-page[data-tm-active='1']")) {
        window.KhatTeachableMachine.load()
            .then(function () {
                return window.KhatTeachableMachine.verifyOnPage("#resultPreviewImg");
            })
            .catch(function () {
                /* model files may be missing locally */
            });
    }
});
