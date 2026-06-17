/* ================================================================
   TomatoAI — Chart module (Chart.js + custom 3D cuboid plugin)
   ================================================================ */

const CHART_COLORS = [
    '#e85d4a', '#f5a623', '#3fb950', '#58a6ff', '#bc8cff',
    '#f778ba', '#f0883e', '#56d4dd', '#db61a2', '#7ee787',
];
const GRID_COLOR  = '#e8ecf1';
const TEXT_COLOR  = '#656d76';
const FLOOR       = 0.025;  // minimum visual bar height (2.5 %)

/* ---------------------------------------------------------------------------
   3D Cuboid Plugin
   ---------------------------------------------------------------------------
   After Chart.js renders the front-face rectangle of each bar, this plugin
   draws a darker right-face parallelogram and a lighter top-face parallelogram
   to create the illusion of a 3-D cuboid (rectangular prism).

   Geometry (isometric-like projection):

         top-left ──── top-right          ← top face (bright)
              ╲          ╲
               ╲          ╲
            front-left  front-right      ← front face (Chart.js default)
                │          │
                │          │
             bottom-left bottom-right

   The plugin only draws the right face (darker) and top face (lighter);
   Chart.js's built-in bar renderer handles the front face.
   --------------------------------------------------------------------------- */
/* Helper: parse "#rrggbb" → [r,g,b] */
function hexToRgb(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return m ? [parseInt(m[1],16), parseInt(m[2],16), parseInt(m[3],16)] : [128,128,128];
}

const cuboid3dPlugin = {
    id: 'cuboid3d',
    afterDatasetDraw(chart) {
        const meta = chart.getDatasetMeta(0);
        if (!meta || !meta.data || meta.data.length === 0) return;

        const { ctx } = chart;
        const bgColors = chart.data.datasets[0].backgroundColor;
        ctx.save();

        const depthX = 12;
        const depthY = 8;

        meta.data.forEach((bar, i) => {
            const [r, g, b] = hexToRgb(bgColors[i] || '#888');
            const L = bar.x - bar.width / 2;
            const R = bar.x + bar.width / 2;
            const T = bar.y;
            const B = bar.base;

            // --- right-face (darker shade of bar color) ---
            ctx.fillStyle = `rgba(${Math.round(r*0.65)},${Math.round(g*0.65)},${Math.round(b*0.65)},1)`;
            ctx.beginPath();
            ctx.moveTo(R, T);
            ctx.lineTo(R + depthX, T - depthY);
            ctx.lineTo(R + depthX, B - depthY);
            ctx.lineTo(R, B);
            ctx.closePath();
            ctx.fill();

            // --- top-face (lighter shade of bar color) ---
            ctx.fillStyle = `rgba(${Math.round(r*1.35 > 255 ? 255 : r*1.35)},${Math.round(g*1.35 > 255 ? 255 : g*1.35)},${Math.round(b*1.35 > 255 ? 255 : b*1.35)},1)`;
            ctx.beginPath();
            ctx.moveTo(L, T);
            ctx.lineTo(R, T);
            ctx.lineTo(R + depthX, T - depthY);
            ctx.lineTo(L + depthX, T - depthY);
            ctx.closePath();
            ctx.fill();

            // --- top edge stroke ---
            ctx.strokeStyle = 'rgba(0,0,0,0.08)';
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(L, T);
            ctx.lineTo(R, T);
            ctx.lineTo(R + depthX, T - depthY);
            ctx.lineTo(L + depthX, T - depthY);
            ctx.closePath();
            ctx.stroke();

            // --- right-edge stroke ---
            ctx.strokeStyle = 'rgba(0,0,0,0.07)';
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(R, T);
            ctx.lineTo(R + depthX, T - depthY);
            ctx.lineTo(R + depthX, B - depthY);
            ctx.stroke();
        });

        ctx.restore();
    },
};

/* ---------------------------------------------------------------------------
   renderProbChart — create a fresh 3D bar chart (top-5 classes, floor applied)
   --------------------------------------------------------------------------- */
let probChart = null;

function renderProbChart(canvasId, labels, values) {
    if (probChart) { probChart.destroy(); probChart = null; }

    // Top-5 by probability
    const combined = labels.map((l, i) => ({ label: l, value: values[i] }));
    combined.sort((a, b) => b.value - a.value);
    const top5Labels   = combined.slice(0, 5).map(c => c.label);
    const originalVals = combined.slice(0, 5).map(c => c.value);
    // Apply floor so near-zero bars remain visible as cuboids
    const displayVals  = originalVals.map(v => Math.max(v, FLOOR));

    const canvas = document.getElementById(canvasId);
    const ctx    = canvas.getContext('2d');

    probChart = new Chart(ctx, {
        type: 'bar',
        plugins: [cuboid3dPlugin],
        data: {
            labels: top5Labels,
            datasets: [{
                label: '置信度',
                data: displayVals,
                backgroundColor: CHART_COLORS.slice(0, 5),
                borderRadius: 0,
                borderSkipped: false,
                barPercentage: 0.68,
                categoryPercentage: 0.76,
                // stash originals for tooltip
                _original: originalVals,
            }],
        },
        options: {
            indexAxis: 'x',
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 1500, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label(ctx) {
                            const orig = ctx.dataset._original[ctx.dataIndex];
                            return ` ${(orig * 100).toFixed(2)}%`;
                        },
                    },
                },
            },
            scales: {
                y: {
                    min: 0, max: 1,
                    ticks: {
                        callback: v => `${(v * 100).toFixed(0)}%`,
                        color: TEXT_COLOR,
                        font: { size: 11 },
                    },
                    grid: { color: GRID_COLOR },
                },
                x: {
                    ticks: {
                        color: TEXT_COLOR,
                        font: { size: 11 },
                        maxRotation: 0,
                    },
                    grid: { display: false },
                },
            },
        },
    });
}

/* ---------------------------------------------------------------------------
   Training curves (model page)
   --------------------------------------------------------------------------- */
let lossChart = null, accChart = null;

const HIST = {
    train_loss: [1.087,0.578,0.480,0.412,0.404,0.292,0.195,0.153,0.124,0.101,0.085,0.071,0.059,0.053,0.047,0.038,0.034,0.032,0.030,0.024,0.025,0.024,0.023,0.019,0.016,0.017,0.014,0.013,0.012,0.013],
    val_loss:   [0.672,0.538,0.472,0.463,0.421,0.299,0.240,0.199,0.174,0.186,0.163,0.152,0.147,0.145,0.129,0.137,0.114,0.103,0.112,0.105,0.094,0.093,0.100,0.105,0.116,0.102,0.092,0.089,0.105,0.090],
    train_acc:  [0.689,0.834,0.854,0.874,0.870,0.909,0.940,0.953,0.962,0.971,0.976,0.980,0.983,0.985,0.988,0.990,0.991,0.992,0.993,0.994,0.994,0.993,0.994,0.996,0.997,0.996,0.996,0.998,0.998,0.997],
    val_acc:    [0.808,0.840,0.848,0.820,0.858,0.906,0.926,0.930,0.942,0.936,0.946,0.950,0.950,0.950,0.960,0.958,0.966,0.966,0.968,0.970,0.972,0.972,0.972,0.972,0.968,0.970,0.970,0.976,0.970,0.970],
};

function createTrainingCharts() {
    if (lossChart) { lossChart.destroy(); lossChart = null; }
    if (accChart)  { accChart.destroy();  accChart  = null; }

    const epochs = Array.from({ length: HIST.train_loss.length }, (_, i) => i + 1);
    const baseOpts = {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 12, usePointStyle: true, color: TEXT_COLOR } } },
        scales: { x: { ticks: { color: TEXT_COLOR, maxTicksLimit: 10 }, grid: { color: GRID_COLOR } }, y: { ticks: { color: TEXT_COLOR }, grid: { color: GRID_COLOR } } },
    };

    const lCtx = document.getElementById('lossChart');
    if (lCtx) {
        lossChart = new Chart(lCtx.getContext('2d'), {
            type: 'line', data: { labels: epochs, datasets: [
                { label:'训练 Loss', data:HIST.train_loss, borderColor:'#e85d4a', backgroundColor:'rgba(232,93,74,.08)', fill:true, tension:.3, pointRadius:0, borderWidth:2 },
                { label:'验证 Loss', data:HIST.val_loss,   borderColor:'#3fb950', backgroundColor:'rgba(63,185,80,.08)',  fill:true, tension:.3, pointRadius:0, borderWidth:2 },
            ]}, options: baseOpts,
        });
    }

    const aCtx = document.getElementById('accChart');
    if (aCtx) {
        accChart = new Chart(aCtx.getContext('2d'), {
            type: 'line', data: { labels: epochs, datasets: [
                { label:'训练 Acc', data:HIST.train_acc, borderColor:'#e85d4a', backgroundColor:'rgba(232,93,74,.08)', fill:true, tension:.3, pointRadius:0, borderWidth:2 },
                { label:'验证 Acc', data:HIST.val_acc,   borderColor:'#3fb950', backgroundColor:'rgba(63,185,80,.08)',  fill:true, tension:.3, pointRadius:0, borderWidth:2 },
            ]}, options: { ...baseOpts,
                scales: { ...baseOpts.scales, y: { ...baseOpts.scales.y, min:.65, max:1, ticks: { ...baseOpts.scales.y.ticks, callback:v=>`${(v*100).toFixed(0)}%` } } },
            },
        });
    }
}
