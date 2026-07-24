/**
 * Perhitungan Algoritma page — print, PDF, smooth scroll.
 */
(function () {
    "use strict";

    function init() {
        var printBtn = document.getElementById("algoPrintBtn");
        var pdfBtn = document.getElementById("algoPdfBtn");

        if (printBtn) {
            printBtn.addEventListener("click", function () {
                window.print();
            });
        }
        if (pdfBtn) {
            pdfBtn.addEventListener("click", function () {
                window.print();
            });
        }

        document.querySelectorAll(".algo-toc-list a[href^='#']").forEach(function (link) {
            link.addEventListener("click", function (event) {
                var id = link.getAttribute("href");
                var target = id ? document.querySelector(id) : null;
                if (target) {
                    event.preventDefault();
                    target.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            });
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
