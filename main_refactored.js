/* eslint-disable no-unused-vars */
/* eslint-disable no-undef */

document.addEventListener("DOMContentLoaded", () => {
    const today = new Date().toISOString().split("T")[0];
    document.getElementById("tanggal").value = today;
    document.getElementById("tgl-awal").value = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
    document.getElementById("tgl-akhir").value = today;
    
    // Panggil fungsi ini saat halaman dimuat
    tampilkanInfoUpdateData();
    document.getElementById('pasaran').addEventListener('change', tampilkanInfoUpdateData);
});

async function callApi(endpoint, method = 'POST', body = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) options.body = JSON.stringify(body);
    try {
        const response = await fetch(endpoint, options);
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Server error: ${response.statusText}`);
        return result;
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
        await callApi('/refresh-data', 'POST', { pasaran });
        alert(`Data untuk pasaran ${pasaran.toUpperCase()} telah berhasil dimuat ulang dari server.`);
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
    if (!confirm(`Anda yakin ingin melatih model baru untuk ${pasaran}? Ini akan memakan waktu.`)) return;

    const statusDiv = document.getElementById('training-status');
    const button = document.getElementById('latih-button');
    const config = {
        learning_rate: parseFloat(document.getElementById('tuning-lr').value),
        epochs: parseInt(document.getElementById('tuning-epochs').value, 10),
        h1: parseInt(document.getElementById('tuning-h1').value, 10),
        h2: parseInt(document.getElementById('tuning-h2').value, 10),
        patience: parseInt(document.getElementById('tuning-patience').value, 10)
    };
    
    button.disabled = true;
    statusDiv.innerHTML = `<div class="spinner"></div><p>Melatih model di server...</p>`;

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
    hasilDiv.innerHTML = `<div class="spinner"></div><p>Memproses evaluasi...</p>`;

    try {
        const result = await callApi('/evaluate', 'POST', { pasaran, tgl_awal, tgl_akhir });
        const summaryHTML = renderSummary(result.summary);
        const matrixHTML = renderConfusionMatrix(result.confusion_matrix);
        const detailsHTML = renderDetails(result.daily_details);
        hasilDiv.innerHTML = `<h4>Ringkasan Akurasi</h4>${summaryHTML}<h4 style="margin-top:30px;">Matriks Konfusi</h4>${matrixHTML}<h4 style="margin-top:30px;">Detail Per Hari</h4>${detailsHTML}`;
    } catch (error) {
        hasilDiv.innerHTML = `<p class="error">${error.message}</p>`;
    } finally {
        button.disabled = false;
    }
}

function renderSummary(summary) {
    const renderRow = (label, data) => `
        <tr>
            <td>${label}</td>
            <td><strong>${(data.accuracy * 100).toFixed(2)}%</strong></td>
            <td>${data.hit}</td>
            <td>${data.miss}</td>
        </tr>`;
    return `<div class="table-container"><table>
        <thead><tr><th>Indikator</th><th>Akurasi</th><th>Hit</th><th>Miss</th></tr></thead>
        <tbody>
            ${renderRow('Colok Bebas (CB)', summary.cb)}
            ${renderRow('Presisi AM (≥1 Angka)', summary.am)}
            ${renderRow('Akurasi Kop (C)', summary.kop)}
            ${renderRow('Akurasi Kepala (K)', summary.kepala)}
            ${renderRow('Akurasi Ekor (E)', summary.ekor)}
        </tbody>
    </table></div>`;
}

function renderConfusionMatrix(matrix) {
    let table = '<div class="table-container"><table><thead><tr><th>Aktual \\ Prediksi</th>';
    for(let i=0; i<10; i++) table += `<th>${i}</th>`;
    table += '</tr></thead><tbody>';
    for(let i=0; i<10; i++) {
        table += `<tr><td><strong>${i}</strong></td>`;
        for(let j=0; j<10; j++) {
            table += `<td class="${i===j ? 'cell-correct' : ''}">${matrix[i][j]}</td>`;
        }
        table += '</tr>';
    }
    table += '</tbody></table></div>';
    return table;
}

function renderDetails(details) {
    let table = '<div class="table-container"><table><thead><tr><th>Tanggal</th><th>Hasil</th><th>CB</th><th>Status</th><th>Angka Main</th><th>Prediksi Posisi</th><th>Status</th></tr></thead><tbody>';
    details.forEach(d => {
        const actualKop = parseInt(d.hasil[1]);
        const actualKepala = parseInt(d.hasil[2]);
        const actualEkor = parseInt(d.hasil[3]);

        const predPosisiParts = d.pred_posisi.split(' '); // Contoh: ["C:1,2,3", "K:4,5,6", "E:7,8,9"]
        
        let formattedPredPosisi = '';
        predPosisiParts.forEach(part => {
            const [posLabel, predNumsStr] = part.split(':'); // Contoh: "C", "1,2,3"
            const predNums = predNumsStr.split(',').map(Number); // Contoh: [1,2,3]
            
            let highlightedNums = predNums.map(num => {
                let isHit = false;
                if (posLabel === 'C' && num === actualKop) isHit = true;
                if (posLabel === 'K' && num === actualKepala) isHit = true;
                if (posLabel === 'E' && num === actualEkor) isHit = true;
                
                return isHit ? `<span class="am-hit">${num}</span>` : num;
            }).join(',');
            formattedPredPosisi += `${posLabel}:${highlightedNums} `;
        });

        const am_highlighted = d.pred_am.map(num => d.am_found.includes(num) ? `<span class="am-hit">${num}</span>` : num).join(', ');
        table += `
            <tr>
                <td>${d.tanggal}</td>
                <td><strong>${d.hasil}</strong></td>
                <td>${d.pred_cb}</td>
                <td class="${d.cb_status.toLowerCase()}">${d.cb_status}</td>
                <td>${am_highlighted}</td>
                <td style="font-size:0.9em;">${formattedPredPosisi.trim()}</td>
                <td class="${d.posisi_status === 'Miss' ? 'miss' : 'hit'}">${d.posisi_status}</td>
            </tr>`;
    });
    table += '</tbody></table></div>';
    return table;
}

async function tampilkanInfoUpdateData() {
    const infoDiv = document.getElementById("data-update-info");
    const pasaran = document.getElementById("pasaran").value;
    infoDiv.textContent = 'Memeriksa pembaruan data...';
    try {
        const result = await callApi(`/get-last-update?pasaran=${pasaran}`, 'GET');
        infoDiv.textContent = `Data historis diperbarui s/d: ${result.last_update}`;
    } catch (error) {
        infoDiv.textContent = 'Gagal memuat info tanggal data.';
        console.error("Gagal memuat info update data:", error);
    }
}