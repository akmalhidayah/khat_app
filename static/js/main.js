document.addEventListener("DOMContentLoaded", () => {
    const alerts = document.querySelectorAll(".alert");
    alerts.forEach((alert) => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            bsAlert.close();
        }, 4500);
    });

    const sidebar = document.getElementById("appSidebar");
    const toggle = document.getElementById("sidebarToggle");
    const collapseBtn = document.getElementById("sidebarCollapseBtn");
    const backdrop = document.getElementById("sidebarBackdrop");
    const sidebarLinks = document.querySelectorAll(".sidebar-link, .sidebar-logout");
    const COLLAPSE_KEY = "khat-sidebar-collapsed";
    const DESKTOP_BREAKPOINT = 992;

    const isDesktop = () => window.innerWidth >= DESKTOP_BREAKPOINT;

    const setSidebarAria = (expanded) => {
        if (toggle) {
            toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
        }
    };

    const closeMobileSidebar = () => {
        document.body.classList.remove("sidebar-open");
        setSidebarAria(false);
    };

    const applyCollapsedState = (collapsed) => {
        document.body.classList.toggle("sidebar-collapsed", collapsed && isDesktop());
        if (collapseBtn) {
            collapseBtn.setAttribute(
                "aria-label",
                collapsed ? "Perluas sidebar" : "Ciutkan sidebar"
            );
            collapseBtn.setAttribute(
                "title",
                collapsed ? "Perluas sidebar" : "Ciutkan sidebar"
            );
        }
    };

    const savedCollapsed = localStorage.getItem(COLLAPSE_KEY) === "1";
    if (savedCollapsed) {
        applyCollapsedState(true);
    }

    if (toggle) {
        toggle.addEventListener("click", () => {
            if (isDesktop()) {
                const nextCollapsed = !document.body.classList.contains("sidebar-collapsed");
                applyCollapsedState(nextCollapsed);
                localStorage.setItem(COLLAPSE_KEY, nextCollapsed ? "1" : "0");
                return;
            }

            const willOpen = !document.body.classList.contains("sidebar-open");
            document.body.classList.toggle("sidebar-open", willOpen);
            setSidebarAria(willOpen);
        });
    }

    if (collapseBtn) {
        collapseBtn.addEventListener("click", () => {
            if (!isDesktop()) return;
            const nextCollapsed = !document.body.classList.contains("sidebar-collapsed");
            applyCollapsedState(nextCollapsed);
            localStorage.setItem(COLLAPSE_KEY, nextCollapsed ? "1" : "0");
        });
    }

    if (backdrop) {
        backdrop.addEventListener("click", closeMobileSidebar);
    }

    sidebarLinks.forEach((link) => {
        link.addEventListener("click", () => {
            if (!isDesktop()) {
                closeMobileSidebar();
            }
        });
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
            closeMobileSidebar();
        }
    });

    window.addEventListener("resize", () => {
        if (!isDesktop()) {
            document.body.classList.remove("sidebar-collapsed");
            closeMobileSidebar();
            return;
        }

        const collapsed = localStorage.getItem(COLLAPSE_KEY) === "1";
        applyCollapsedState(collapsed);
        closeMobileSidebar();
    });

    const topbar = document.getElementById("appTopbar");
    const scrollRoot = document.querySelector(".main-content");
    if (topbar && scrollRoot) {
        const updateTopbarScroll = () => {
            topbar.classList.toggle("is-sticky-scrolled", scrollRoot.scrollTop > 10);
        };
        updateTopbarScroll();
        scrollRoot.addEventListener("scroll", updateTopbarScroll, { passive: true });
    }
});
