/* ================================================================
   TomatoAI — Main application logic
   ================================================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ---- DOM refs ----
    const sidebar    = document.getElementById('sidebar');
    const navItems   = sidebar.querySelectorAll('.nav-item');
    const pages      = document.querySelectorAll('.page');

    // Predict page
    const dropZone   = document.getElementById('dropZone');
    const fileInput  = document.getElementById('fileInput');
    const previewImg = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn   = document.getElementById('clearBtn');
    const placeholder = document.getElementById('resultsPlaceholder');
    const resultsContent = document.getElementById('resultsContent');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const badgeValue = document.getElementById('badgeValue');
    const badgeConf   = document.getElementById('badgeConfidence');
    const diseaseCard = document.getElementById('diseaseCard');
    const diseaseTitle = document.getElementById('diseaseTitle');
    const diseaseDesc  = document.getElementById('diseaseDesc');
    const treatmentList = document.getElementById('treatmentList');

    let selectedFile = null;

    // ---- Navigation ----
    navItems.forEach(btn => {
        btn.addEventListener('click', () => {
            const pageId = btn.dataset.page;
            navItems.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            pages.forEach(p => p.classList.remove('active'));
            document.getElementById(`page-${pageId}`).classList.add('active');

            // Initialize training charts when switching to model page
            if (pageId === 'model') {
                setTimeout(createTrainingCharts, 150);
            }
        });
    });

    // ---- File input / drag & drop ----
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            previewImg.classList.remove('hidden');
            dropZone.classList.add('has-image');
            dropZone.querySelector('.drop-content').style.display = 'none';
        };
        reader.readAsDataURL(file);
        analyzeBtn.disabled = false;
        // Reset results
        resetResults();
    }

    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewImg.src = '';
        previewImg.classList.add('hidden');
        dropZone.classList.remove('has-image');
        dropZone.querySelector('.drop-content').style.display = '';
        analyzeBtn.disabled = true;
        resetResults();
    });

    // ---- Predict ----
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        showLoading();
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const resp = await fetch('/api/predict', { method: 'POST', body: formData });
            if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
            const data = await resp.json();
            renderResults(data);
        } catch (err) {
            console.error(err);
            alert('诊断失败，请确认服务器已启动。\n' + err.message);
            hideLoading();
        }
    });

    function showLoading() {
        placeholder.classList.add('hidden');
        resultsContent.classList.add('hidden');
        diseaseCard.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
    }
    function hideLoading() {
        loadingSpinner.classList.add('hidden');
    }

    function resetResults() {
        placeholder.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        diseaseCard.classList.add('hidden');
        loadingSpinner.classList.add('hidden');
    }

    function renderResults(data) {
        hideLoading();
        placeholder.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        // Badge
        badgeValue.textContent = data.predicted_cn;
        badgeConf.textContent = `置信度 ${(data.confidence * 100).toFixed(1)}%`;

        // Bar chart — create with real data (avoids empty-dataset init bug)
        renderProbChart('probChart', data.cn_classes, data.probabilities);

        // Disease info — appears below upload on the left
        const info = data.disease_info;
        diseaseTitle.textContent = info.cn;
        diseaseDesc.textContent = info.desc;
        treatmentList.innerHTML = info.treatment.map(t => `<li>${t}</li>`).join('');
        diseaseCard.classList.remove('hidden');
    }

}); // DOMContentLoaded
