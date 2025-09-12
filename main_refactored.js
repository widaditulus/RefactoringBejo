/* eslint-disable no-unused-vars */
/* eslint-disable no-undef */

// Variabel global untuk menyimpan state bobot
let currentPasaranWeights = {};

document.addEventListener("DOMContentLoaded", () => {
    const today = new Date().toISOString().split("T")[0];
    document.getElementById("tanggal").value = today;
    document.getElementById("tgl-awal").value = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
    document.getElementById("tgl-akhir").value = today;
    
    tampilkanInfoUpdateData();
    document.getElementById('pasaran').addEventListener('change', tampilkanInfoUpdateData);

    // Setup untuk panel tuning bobot baru
    const pasaranTuningSelect = document.getElementById('pasaran-tuning');
    createSliders();
    loadWeightsForPasaran(); // Muat bobot untuk pasaran default saat halaman dibuka
    pasaranTuningSelect.addEventListener('change', loadWeightsForPasaran);
});

async function callApi(endpoint, method = 'GET', body = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) options.body = JSON.stringify(body);
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            let errorText = `Server error: ${response.status} ${response.statusText}`;
            try {
                const result = await response.json();
                errorText = result.error || errorText;
            } catch (e) { /* Abaikan jika respons bukan JSON */ }
            throw new Error(errorText);
        }
        return await response.json();
    } catch (error) {
        console.error(`Kesalahan pada endpoint ${endpoint}:`, error);
        throw error;
    }
}

async function prediksiCB() {
    const pasaran = document.getElementById("pasaran").value;
    const tanggal = document.getElementById("tanggal").value;
    const hasilDiv = document.getElementById("hasil");
    const button = document.querySelector('button[onclick="prediksiCB()"]');
    
    button.disabled = true;
    hasilDiv.style.display = 'block';
    hasilDiv.innerHTML = `<div class="spinner"></div><p>Memproses prediksi...</p>`;

    try {
        const result = await callApi('/predict', 'POST', { pasaran, tanggal });
        const amString = result.am.join(', ');
        const posisiHTML = `
            <p><strong>As:</strong> ${result.posisi.as.join(', ')}</p>
            <p><strong>Kop:</strong> ${result.posisi.kop.join(', ')}</p>
            <p><strong>Kepala:</strong> ${result.posisi.kepala.join(', ')}</p>
            <p><strong>Ekor:</strong> ${result.posisi.ekor.join(', ')}</p>`;
        hasilDiv.innerHTML = `
            <h2>Prediksi CB: <strong>${result.cb}</strong></h2>
            <h3>Angka Main (AM): <strong>${amString}</strong></h3>
            <div style="text-align:left; display:inline-block;">${posisiHTML}</div>`;
    } catch (error) {
        hasilDiv.innerHTML = `<p class="error">${error.message}</p>`;
    } finally {
        button.disabled = false;
    }
}

async function refreshData(buttonElement) {
    const pasaran = document.getElementById("pasaran").value;
    const originalText = buttonElement.innerHTML;

    buttonElement.disabled = true;
    buttonElement.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Memuat...';

    try {
        const result = await callApi('/refresh-data', 'POST', { pasaran });
        alert(result.message || `Data untuk pasaran ${pasaran.toUpperCase()} telah berhasil dimuat ulang.`);
        await tampilkanInfoUpdateData(); 
    } catch (error) {
        alert(`Gagal memuat ulang data: ${error.message}`);
    } finally {
        buttonElement.disabled = false;
        buttonElement.innerHTML = originalText;
    }
}

async function latihModelTerpilih() {
    const pasaran = document.getElementById('pasaran-train').value;
    const button = document.getElementById('latih-button');
    
    if (button.disabled) return;
    if (!confirm(`Anda yakin ingin melatih model baru untuk ${pasaran}? Ini akan menimpa model lama.`)) return;

    const statusDiv = document.getElementById('training-status');
    
    const getConfigValues = (idPrefix) => {
        const epochs = document.getElementById(`tuning-epochs-${idPrefix}`).value;
        const h1 = document.getElementById(`tuning-h1-${idPrefix}`).value;
        const h2 = document.getElementById(`tuning-h2-${idPrefix}`).value;
        const patience = document.getElementById(`tuning-patience-${idPrefix}`).value;
        
        const config = {};
        if (epochs) config.epochs = parseInt(epochs, 10);
        if (h1) config.h1 = parseInt(h1, 10);
        if (h2) config.h2 = parseInt(h2, 10);
        if (patience) config.patience = parseInt(patience, 10);
        return config;
    };

    const config = {
        global: getConfigValues('global'),
        pos_specific: {} // Anda bisa menambahkan UI untuk ini jika diperlukan
    };
    
    button.disabled = true;
    statusDiv.innerHTML = `<div class="spinner"></div><p>Melatih model di server (sekitar 5-10 detik)...</p>`;

    try {
        const result = await callApi('/train', 'POST', { pasaran, config });
        statusDiv.innerHTML = `<p style="color: green; font-weight: bold;">${result.message}</p>`;
        alert(result.message);
    } catch (error) {
        statusDiv.innerHTML = `<p class="error">${error.message}</p>`;
    } finally {
        button.disabled = false;
    }
}

async function updateWeights() {
    const pasaran = document.getElementById('pasaran-train').value;
    if (!confirm(`Anda yakin ingin menjalankan pembelajaran adaptif untuk ${pasaran}? Bobot pada panel tuning di bawah akan diperbarui.`)) return;

    const statusDiv = document.getElementById('training-status');
    const button = document.getElementById('update-weights-button');
    button.disabled = true;
    statusDiv.innerHTML = `<div class="spinner"></div><p>Menganalisis data & memperbarui bobot...</p>`;

    try {
        const startResult = await callApi('/update-weights', 'POST', { pasaran });
        const { job_id } = startResult;

        const intervalId = setInterval(async () => {
            try {
                const jobStatus = await callApi(`/update-weights-status/${job_id}`);
                if (jobStatus.status === 'complete' || jobStatus.status === 'error') {
                    clearInterval(intervalId);
                    button.disabled = false;
                    if (jobStatus.status === 'complete') {
                        statusDiv.innerHTML = `<p style="color: blue; font-weight: bold;">${jobStatus.message}</p>`;
                        if (pasaran === document.getElementById('pasaran-tuning').value) {
                            loadWeightsForPasaran();
                        }
                    } else {
                        throw new Error(jobStatus.message);
                    }
                }
            } catch (pollError) {
                clearInterval(intervalId);
                button.disabled = false;
                statusDiv.innerHTML = `<p class="error">Gagal memeriksa status: ${pollError.message}</p>`;
            }
        }, 2000);
    } catch (startError) {
        statusDiv.innerHTML = `<p class="error">${startError.message}</p>`;
        button.disabled = false;
    }
}

function renderEvaluationResults(jobStatus) {
    const hasilDiv = document.getElementById('hasil-evaluasi');
    const { summary } = jobStatus;
    if (!summary) {
        hasilDiv.innerHTML = `<p class="error">Evaluasi selesai, namun tidak ada data ringkasan.</p>`;
        return;
    }
    let summaryHtml = `<h4>Ringkasan Akurasi</h4><div class="table-container"><table><thead><tr><th>INDIKATOR</th><th>AKURASI</th><th>HIT</th><th>MISS</th></tr></thead><tbody>`;
    const indicators = {cb: 'Colok Bebas (CB)', am: 'Presisi AM (≥1 Angka)', as: 'Akurasi As (A)', kop: 'Akurasi Kop (C)', kepala: 'Akurasi Kepala (K)', ekor: 'Akurasi Ekor (E)'};
    for (const key in indicators) {
        if (summary[key]) {
            summaryHtml += `<tr><td>${indicators[key]}</td><td>${summary[key].accuracy.toFixed(2)}%</td><td>${summary[key].hit}</td><td>${summary[key].miss}</td></tr>`;
        }
    }
    summaryHtml += `</tbody></table></div>`;
    hasilDiv.innerHTML = summaryHtml;
}

async function evalKinerja() {
    const pasaran = document.getElementById('pasaran-eval').value;
    const tgl_awal = document.getElementById('tgl-awal').value;
    const tgl_akhir = document.getElementById('tgl-akhir').value;
    const hasilDiv = document.getElementById('hasil-evaluasi');
    const button = document.querySelector('button[onclick="evalKinerja()"]');

    if (!tgl_awal || !tgl_akhir) {
        hasilDiv.innerHTML = `<p class="error">Tanggal Awal dan Akhir harus diisi.</p>`;
        return;
    }
    button.disabled = true;
    hasilDiv.innerHTML = `<div class="spinner"></div><p>Memulai proses evaluasi di server...</p>`;
    try {
        const startResult = await callApi('/evaluate', 'POST', { pasaran, tgl_awal, tgl_akhir });
        const { job_id } = startResult;
        const intervalId = setInterval(async () => {
            try {
                const jobStatus = await callApi(`/evaluate-status/${job_id}`);
                if (jobStatus.status === 'complete' || jobStatus.status === 'error') {
                    clearInterval(intervalId);
                    button.disabled = false;
                    if (jobStatus.status === 'complete') {
                        renderEvaluationResults(jobStatus);
                    } else { throw new Error(jobStatus.message); }
                }
            } catch (pollError) {
                clearInterval(intervalId);
                button.disabled = false;
                hasilDiv.innerHTML = `<p class="error">Gagal memeriksa status: ${pollError.message}</p>`;
            }
        }, 2000);
    } catch (startError) {
        hasilDiv.innerHTML = `<p class="error">${startError.message}</p>`;
        button.disabled = false;
    }
}

async function tampilkanInfoUpdateData() {
    const infoDiv = document.getElementById("data-update-info");
    const pasaran = document.getElementById("pasaran").value;
    infoDiv.textContent = 'Memeriksa pembaruan data...';
    try {
        const result = await callApi(`/get-last-update?pasaran=${pasaran}`);
        infoDiv.textContent = `Data historis diperbarui s/d: ${result.last_update}`;
    } catch (error) {
        infoDiv.textContent = `Gagal memuat info tanggal data: ${error.message}`;
    }
}

function openTab(evt, tabName) {
    let i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// =======================================================
// ## FUNGSI BARU UNTUK TUNING BOBOT
// =======================================================

function createSliders() {
    const container = document.getElementById('sliders-container');
    container.innerHTML = '';
    const positions = { as: 'AS', kop: 'KOP', kepala: 'KEPALA', ekor: 'EKOR' };
    for (const key in positions) {
        const el = document.createElement('div');
        el.className = 'tuning-slider-container';
        el.innerHTML = `
            <label>${positions[key]}</label>
            <div class="slider-labels">
                <span class="ai-label">AI: <b id="label-ai-${key}">50%</b></span>
                <span class="freq-label">Frekuensi: <b id="label-freq-${key}">50%</b></span>
            </div>
            <input type="range" id="slider-${key}" min="0" max="100" value="50">
        `;
        container.appendChild(el);
        document.getElementById(`slider-${key}`).addEventListener('input', (e) => {
            const aiValue = e.target.value;
            document.getElementById(`label-ai-${key}`).textContent = `${aiValue}%`;
            document.getElementById(`label-freq-${key}`).textContent = `${100 - aiValue}%`;
        });
    }
}

async function loadWeightsForPasaran() {
    const pasaran = document.getElementById('pasaran-tuning').value;
    try {
        const weights = await callApi(`/get-weights?pasaran=${pasaran}`);
        currentPasaranWeights = JSON.parse(JSON.stringify(weights)); // Simpan salinan sebagai default
        
        for (const pos in weights) {
            const aiPercent = Math.round(weights[pos].ai * 100);
            document.getElementById(`slider-${pos}`).value = aiPercent;
            document.getElementById(`label-ai-${pos}`).textContent = `${aiPercent}%`;
            document.getElementById(`label-freq-${pos}`).textContent = `${100 - aiPercent}%`;
        }
    } catch (error) {
        alert(`Gagal memuat bobot untuk ${pasaran}: ${error.message}`);
    }
}

function resetWeights() {
    if (Object.keys(currentPasaranWeights).length === 0) {
        alert("Data default belum dimuat. Coba ganti pasaran terlebih dahulu.");
        return;
    }
    for (const pos in currentPasaranWeights) {
        const aiPercent = Math.round(currentPasaranWeights[pos].ai * 100);
        document.getElementById(`slider-${pos}`).value = aiPercent;
        document.getElementById(`label-ai-${pos}`).textContent = `${aiPercent}%`;
        document.getElementById(`label-freq-${pos}`).textContent = `${100 - aiPercent}%`;
    }
}

async function saveWeights() {
    const pasaran = document.getElementById('pasaran-tuning').value;
    const newWeights = {};
    const positions = ['as', 'kop', 'kepala', 'ekor'];

    positions.forEach(pos => {
        const aiPercent = parseInt(document.getElementById(`slider-${pos}`).value, 10);
        newWeights[pos] = {
            ai: aiPercent / 100.0,
            freq: (100 - aiPercent) / 100.0
        };
    });

    try {
        const result = await callApi('/save-weights', 'POST', { pasaran, weights: newWeights });
        alert(result.message);
    } catch (error) {
        alert(`Gagal menyimpan bobot: ${error.message}`);
    }
}

// ## FUNGSI UNTUK MENAMPILKAN/MENYEMBUNYIKAN PANDUAN TUNING
function toggleTuningGuide() {
    const content = document.getElementById('tuning-guide-content');
    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
    } else {
        content.style.display = 'none';
    }
}