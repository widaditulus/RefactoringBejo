/* eslint-disable no-unused-vars */
/* eslint-disable no-undef */

document.addEventListener("DOMContentLoaded", () => {
    const today = new Date().toISOString().split("T")[0];
    document.getElementById("tanggal").value = today;
    document.getElementById("tgl-awal").value = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
    document.getElementById("tgl-akhir").value = today;
    
    tampilkanInfoUpdateData();
    document.getElementById('pasaran').addEventListener('change', tampilkanInfoUpdateData);
});

async function callApi(endpoint, method = 'POST', body = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) options.body = JSON.stringify(body);
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            let errorText = `Server error: ${response.status} ${response.statusText}`;
            try {
                const result = await response.json();
                errorText = result.error || errorText;
            } catch (e) {
                // Abaikan jika respons bukan JSON
            }
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

    // Mencegah klik ganda seketika
    if (button.disabled) {
        return; // Hentikan jika tombol sudah nonaktif
    }
    
    if (!confirm(`Anda yakin ingin melatih model baru untuk ${pasaran}? Ini akan memakan waktu.`)) return;

    const statusDiv = document.getElementById('training-status');
    
    const getConfigValues = (idPrefix) => {
        const lr = document.getElementById(`tuning-lr-${idPrefix}`).value;
        const epochs = document.getElementById(`tuning-epochs-${idPrefix}`).value;
        const h1 = document.getElementById(`tuning-h1-${idPrefix}`).value;
        const h2 = document.getElementById(`tuning-h2-${idPrefix}`).value;
        const patience = document.getElementById(`tuning-patience-${idPrefix}`).value;
        
        const config = {};
        // Hanya tambahkan jika nilainya ada, untuk menghindari mengirim string kosong
        if (lr) config.learning_rate = parseFloat(lr);
        if (epochs) config.epochs = parseInt(epochs, 10);
        if (h1) config.h1 = parseInt(h1, 10);
        if (h2) config.h2 = parseInt(h2, 10);
        if (patience) config.patience = parseInt(patience, 10);
        return config;
    };

    const config = {
        global: getConfigValues('global'),
        pos_specific: {
            as: getConfigValues('as'),
            kop: getConfigValues('kop'),
            kepala: getConfigValues('kepala'),
            ekor: getConfigValues('ekor')
        }
    };
    
    button.disabled = true;
    button.style.cursor = 'not-allowed'; // Umpan balik visual langsung
    statusDiv.innerHTML = `<div class="spinner"></div><p>Melatih model di server (bisa memakan waktu beberapa menit)...</p>`;

    try {
        const result = await callApi('/train', 'POST', { pasaran, config });
        statusDiv.innerHTML = `<p style="color: green; font-weight: bold;">${result.message}</p>`;
        alert(result.message);
    } catch (error) {
        statusDiv.innerHTML = `<p class="error">${error.message}</p>`;
    } finally {
        button.disabled = false;
        button.style.cursor = 'pointer'; // Kembalikan cursor seperti semula
    }
}

async function updateWeights() {
    const pasaran = document.getElementById('pasaran-train').value;
    if (!confirm(`Anda yakin ingin menjalankan pembelajaran adaptif untuk ${pasaran}? Ini akan menganalisis data historis dan memperbarui bobot prediksi.`)) return;

    const statusDiv = document.getElementById('training-status');
    const button = document.getElementById('update-weights-button');
    const progressDiv = document.getElementById('training-progress-indicator');
    const progressText = document.getElementById('training-progress-text');
    const progressBar = document.getElementById('training-progress-bar');

    button.disabled = true;
    statusDiv.innerHTML = '';
    progressDiv.style.display = 'block';
    progressText.textContent = 'Memulai proses di server...';
    progressBar.style.width = '0%';

    try {
        const startResult = await callApi('/update-weights', 'POST', { pasaran });
        const { job_id } = startResult;

        const intervalId = setInterval(async () => {
            try {
                const jobStatus = await callApi(`/update-weights-status/${job_id}`, 'GET');

                progressText.textContent = jobStatus.message || 'Memproses...';
                progressBar.style.width = `${jobStatus.progress || 0}%`;

                if (jobStatus.status === 'complete' || jobStatus.status === 'error') {
                    clearInterval(intervalId);
                    button.disabled = false;
                    progressDiv.style.display = 'none';

                    if (jobStatus.status === 'complete') {
                        statusDiv.innerHTML = `<p style="color: blue; font-weight: bold;">${jobStatus.message}</p>`;
                    } else {
                        throw new Error(jobStatus.message);
                    }
                }
            } catch (pollError) {
                clearInterval(intervalId);
                button.disabled = false;
                progressDiv.style.display = 'none';
                const errorMessage = pollError instanceof Error ? pollError.message : String(pollError);
                statusDiv.innerHTML = `<p class="error">Gagal memeriksa status: ${errorMessage}</p>`;
            }
        }, 2000);

    } catch (startError) {
        progressDiv.style.display = 'none';
        statusDiv.innerHTML = `<p class="error">${startError.message}</p>`;
        button.disabled = false;
    }
}


function renderEvaluationResults(jobStatus) {
    const hasilDiv = document.getElementById('hasil-evaluasi');
    const { summary, daily_details, confusion_matrix } = jobStatus;

    // 1. Ringkasan Akurasi
    let summaryHtml = `
        <h4>Ringkasan Akurasi</h4>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>INDIKATOR</th>
                        <th>AKURASI</th>
                        <th>HIT</th>
                        <th>MISS</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Colok Bebas (CB)</td>
                        <td>${summary.cb.accuracy.toFixed(2)}%</td>
                        <td>${summary.cb.hit}</td>
                        <td>${summary.cb.miss}</td>
                    </tr>
                    <tr>
                        <td>Presisi AM (≥1 Angka)</td>
                        <td>${summary.am.accuracy.toFixed(2)}%</td>
                        <td>${summary.am.hit}</td>
                        <td>${summary.am.miss}</td>
                    </tr>
                    <tr>
                        <td>Akurasi As (A)</td>
                        <td>${summary.as.accuracy.toFixed(2)}%</td>
                        <td>${summary.as.hit}</td>
                        <td>${summary.as.miss}</td>
                    </tr>
                     <tr>
                        <td>Akurasi Kop (C)</td>
                        <td>${summary.kop.accuracy.toFixed(2)}%</td>
                        <td>${summary.kop.hit}</td>
                        <td>${summary.kop.miss}</td>
                    </tr>
                     <tr>
                        <td>Akurasi Kepala (K)</td>
                        <td>${summary.kepala.accuracy.toFixed(2)}%</td>
                        <td>${summary.kepala.hit}</td>
                        <td>${summary.kepala.miss}</td>
                    </tr>
                     <tr>
                        <td>Akurasi Ekor (E)</td>
                        <td>${summary.ekor.accuracy.toFixed(2)}%</td>
                        <td>${summary.ekor.hit}</td>
                        <td>${summary.ekor.miss}</td>
                    </tr>
                </tbody>
            </table>
        </div>`;

    // 2. Matriks Konfusi
    let matrixHtml = `
        <h4 style="margin-top: 30px;">Matriks Konfusi (Prediksi CB)</h4>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th rowspan="2" style="vertical-align: middle;">AKTUAL</th>
                        <th colspan="10">PREDIKSI</th>
                    </tr>
                    <tr>
                        ${Array.from(Array(10).keys()).map(i => `<th>${i}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>`;
    
    confusion_matrix.forEach((row, i) => {
        matrixHtml += `<tr><th>${i}</th>`;
        row.forEach((cell, j) => {
            const style = i === j ? 'background-color: #d5f5e3; color: #1e8449; font-weight: bold;' : '';
            matrixHtml += `<td style="${style}">${cell}</td>`;
        });
        matrixHtml += `</tr>`;
    });
    matrixHtml += `</tbody></table></div>`;

    // 3. Detail Evaluasi Per Hari
    let detailsHtml = `
        <h4 style="margin-top: 30px;">Detail Evaluasi Per Hari</h4>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th rowspan="2" style="vertical-align: middle;">TANGGAL</th>
                        <th rowspan="2" style="vertical-align: middle;">HASIL</th>
                        <th colspan="2">COLOK BEBAS</th>
                        <th rowspan="2" style="vertical-align: middle;">ANGKA MAIN</th>
                        <th rowspan="2" style="vertical-align: middle;">PREDIKSI POSISI (KANDIDAT)</th>
                        <th rowspan="2" style="vertical-align: middle;">STATUS</th>
                    </tr>
                    <tr>
                        <th>PREDIKSI</th>
                        <th>STATUS</th>
                    </tr>
                </thead>
                <tbody>`;

    daily_details.forEach(day => {
        const cbStatusStyle = day.cb_status === 'Hit' ? 'color: green; font-weight: bold;' : 'color: red;';
        const finalStatusStyle = day.final_status === 'Miss' ? 'color: red;' : 'color: green; font-weight: bold;';
        
        const hitStyle = 'color: red; font-weight: bold;';
        const actualDigitsSet = new Set(day.hasil.split(''));

        const cbHtml = actualDigitsSet.has(String(day.pred_cb)) 
            ? `<span style="${hitStyle}">${day.pred_cb}</span>` 
            : day.pred_cb;

        const amHtml = day.pred_am.map(num => 
            actualDigitsSet.has(String(num)) 
                ? `<span style="${hitStyle}">${num}</span>` 
                : num
        ).join(', ');

        const actualPosDigits = {
            A: day.hasil[0],
            C: day.hasil[1],
            K: day.hasil[2],
            E: day.hasil[3]
        };
        
        const posHtml = day.pos_kandidat.split(' ').map(part => {
            const [label, numbersStr] = part.split(':');
            if (!numbersStr) return part;

            const actualForPos = actualPosDigits[label];
            const highlightedNumbers = numbersStr.split(',').map(num =>
                num === actualForPos ? `<span style="${hitStyle}">${num}</span>` : num
            ).join(',');
            
            return `${label}: ${highlightedNumbers}`;
        }).join('&nbsp;&nbsp;'); 
        
        detailsHtml += `
            <tr>
                <td>${day.tanggal}</td>
                <td>${day.hasil}</td>
                <td>${cbHtml}</td>
                <td style="${cbStatusStyle}">${day.cb_status}</td>
                <td>${amHtml}</td>
                <td>${posHtml}</td>
                <td style="${finalStatusStyle}">${day.final_status}</td>
            </tr>`;
    });
    detailsHtml += `</tbody></table></div>`;

    hasilDiv.innerHTML = summaryHtml + matrixHtml + detailsHtml;
}


async function evalKinerja() {
    const pasaran = document.getElementById('pasaran-eval').value;
    const tgl_awal = document.getElementById('tgl-awal').value;
    const tgl_akhir = document.getElementById('tgl-akhir').value;
    const hasilDiv = document.getElementById('hasil-evaluasi');
    const progressDiv = document.getElementById('eval-progress-indicator');
    const progressText = document.getElementById('eval-progress-text');
    const progressBar = document.getElementById('eval-progress-bar');
    const button = document.querySelector('button[onclick="evalKinerja()"]');

    if (!tgl_awal || !tgl_akhir) {
        hasilDiv.innerHTML = `<p class="error">Tanggal Awal dan Akhir harus diisi.</p>`;
        return;
    }

    button.disabled = true;
    hasilDiv.innerHTML = '';
    progressDiv.style.display = 'block';
    progressText.textContent = 'Memulai proses evaluasi di server...';
    progressBar.style.width = '0%';

    try {
        const startResult = await callApi('/evaluate', 'POST', { pasaran, tgl_awal, tgl_akhir });
        const { job_id } = startResult;

        const intervalId = setInterval(async () => {
            try {
                const jobStatus = await callApi(`/evaluate-status/${job_id}`, 'GET');

                progressText.textContent = jobStatus.message || 'Memproses...';
                progressBar.style.width = `${jobStatus.progress || 0}%`;

                if (jobStatus.status === 'complete' || jobStatus.status === 'error') {
                    clearInterval(intervalId);
                    button.disabled = false;
                    progressDiv.style.display = 'none';

                    if (jobStatus.status === 'complete') {
                        renderEvaluationResults(jobStatus);
                    } else {
                        throw new Error(jobStatus.message);
                    }
                }
            } catch (pollError) {
                clearInterval(intervalId);
                button.disabled = false;
                progressDiv.style.display = 'none';
                const errorMessage = pollError instanceof Error ? pollError.message : String(pollError);
                hasilDiv.innerHTML = `<p class="error">Gagal memeriksa status: ${errorMessage}</p>`;
            }
        }, 2000);

    } catch (startError) {
        progressDiv.style.display = 'none';
        hasilDiv.innerHTML = `<p class="error">${startError.message}</p>`;
        button.disabled = false;
    }
}


async function tampilkanInfoUpdateData() {
    const infoDiv = document.getElementById("data-update-info");
    const pasaran = document.getElementById("pasaran").value;
    infoDiv.textContent = 'Memeriksa pembaruan data...';
    try {
        const result = await callApi(`/get-last-update?pasaran=${pasaran}`, 'GET');
        infoDiv.textContent = `Data historis diperbarui s/d: ${result.last_update}`;
    } catch (error) {
        infoDiv.textContent = `Gagal memuat info tanggal data: ${error.message}`;
        console.error("Gagal memuat info update data:", error);
    }
}

function openTab(evt, tabName) {
    let i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

function toggleTuningGuide() {
    const content = document.getElementById('tuning-guide-content');
    if (content.style.display === 'none' || content.style.display === '') {
        content.style.display = 'block';
    } else {
        content.style.display = 'none';
    }
}