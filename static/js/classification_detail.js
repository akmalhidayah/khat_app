document.addEventListener("DOMContentLoaded", function () {
    const chartData = window.ALGO_DETAIL_CHART;
    const canvas = document.getElementById("probChart");
    if (!canvas || !chartData || !chartData.labels || !chartData.labels.length) {
        return;
    }

    const colors = ["#0d9488", "#0284c7", "#d97706", "#7c3aed", "#dc2626", "#64748b"];
    const bgColors = chartData.labels.map(function (_, i) {
        return colors[i % colors.length] + "cc";
    });

    new Chart(canvas, {
        type: "bar",
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: "Probabilitas (%)",
                    data: chartData.values,
                    backgroundColor: bgColors,
                    borderColor: colors.slice(0, chartData.labels.length),
                    borderWidth: 1,
                    borderRadius: 6,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            return ctx.parsed.y.toFixed(2) + "%";
                        },
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function (v) {
                            return v + "%";
                        },
                    },
                    title: { display: true, text: "Persentase" },
                },
                x: {
                    ticks: { maxRotation: 45, minRotation: 0 },
                },
            },
        },
    });
});
