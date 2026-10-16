# Model Challenger: LSTM, GRU, ARIMA-GARCH, TFT

Empat model tambahan yang dijalankan **berdampingan** dengan model aktif tanpa mengganggunya.

| Yang TIDAK berubah | Yang baru |
|---|---|
| Model aktif (`model_versions`), sinyal BUY/WAIT/AVOID, ranking, risk engine | Tabel `challenger_models` (registry, metrik, artifact) |
| Workflow Daily, Setup, Retrain, Backtest, Tests | Tabel `model_forecasts` (prakiraan harian per model) |
| `requirements.txt` (produksi tidak memasang torch) | Workflow `challenger_tests.yml`, `challenger_train.yml`, `challenger_predict.yml` |
| | `requirements-models.txt`, `config/challengers.yaml`, paket `app/challengers/`, halaman dashboard **Model Lab** |

Challenger **tidak pernah** memengaruhi keputusan transaksi. Promosi ke model utama hanya boleh lewat keputusan
eksplisit setelah bukti out-of-sample yang konsisten (lihat "Kapan challenger boleh dipakai").

## Apa yang diprakirakan

Semua model memprakirakan **distribusi log return kumulatif 5 hari bursa** (`labels.SWING_HORIZON`) untuk setiap
emiten, lalu diturunkan menjadi besaran yang sama:

| Kolom | Arti |
|---|---|
| `exp_return` | ekspektasi return sederhana 5 hari |
| `prob_up` | P(return 5 hari > 0) |
| `prob_bullish` / `prob_bearish` | P(return > +3%) / P(return < −3%), ambang yang sama dengan label `y_class` model aktif |
| `vol` | σ log return 5 hari |
| `q10`, `q50`, `q90` | kuantil return 5 hari |

## Model

### ARIMA-GARCH (per emiten, statistik klasik)
- Mean: ARIMA(p,0,q) pada log return harian (setara ARIMA(p,1,q) pada log harga). Orde dipilih per emiten dengan AIC
  dari grid `(p,q) ∈ {(0,0),(1,0),(0,1),(1,1),(2,0),(0,2)}`.
- Varians: GARCH(1,1) dengan inovasi Student-t pada residual ARMA (estimasi dua tahap: `statsmodels` lalu `arch`).
- Prakiraan 5 hari: mean = Σ E[r_{t+k}], varians = Σ σ²_{t+k|t}·(Σψ_j)², dengan ψ = bobot MA(∞). Distribusi jumlah
  didekati normal. Bila GARCH gagal atau tidak stasioner, dipakai fallback EWMA (λ = 0,94) yang ditandai.
- Rekursi dihitung dengan numpy (`scipy.signal.lfilter`) memakai parameter tetap. Kebenarannya dibuktikan secara
  analitik, dengan Monte Carlo, dan dengan perbandingan langsung terhadap `statsmodels` dan `arch` di CI.

### LSTM dan GRU (global, sekuens)
- Input: 60 hari terakhir dari ≤ 64 fitur `f_*` (yang sama dengan model aktif, dipilih berdasarkan cakupan data
  training) ditambah return harian, return 5 hari, dan perubahan volume.
- 2 lapis, hidden 64, kepala kuantil monoton (7 kuantil, pinball loss).

### TFT (Temporal Fusion Transformer)
Implementasi sendiri di PyTorch mengikuti Lim et al. (2021). `pytorch-forecasting` tidak dipakai karena dependensinya
berat dan rawan konflik dengan pandas 3 yang dipakai repo ini. Komponennya:
- static covariate encoder (sektor);
- variable selection network untuk input masa lalu dan masa depan yang diketahui (hari dalam minggu);
- encoder–decoder LSTM;
- static enrichment;
- interpretable multi-head attention (kausal);
- kuantil untuk horizon 1..5.

Bobot pemilihan variabel dan attention bisa dibaca untuk interpretasi.

## Evaluasi yang adil

| Aspek | Sama dengan model aktif |
|---|---|
| Label | `y_ret_5d`, `y_class` (±3%) dari `app/models/labels.py` |
| Fitur | `build_dataset` yang sama (bebas look-ahead; diuji `test_no_data_leakage`) |
| Periode test | fold `test` dari `make_folds` (6 bulan terakhir) dengan embargo 10 hari |
| Baris yang dinilai | `y_class` tersedia **dan** `is_tradeable` |
| Metrik | `roc_auc_bullish`, `log_loss`, `ece_bullish`, `topk_mean_net_return` (top-5/hari, setelah biaya round-trip) |

Tambahan untuk prakiraan distribusi:
- IC harian (korelasi peringkat antar-emiten), dengan t-stat yang dikoreksi kasar karena target saling bertumpuk;
- AUC arah;
- cakupan interval 80% (ideal 0,80);
- pinball loss;
- korelasi σ prakiraan dengan |return| aktual.

Pembanding pada baris yang sama: baseline momentum (`f_roc10`), reversal (kebalikannya), dan rata-rata universe.

### Alur pelatihan (workflow mingguan, Sabtu 11:37 UTC = 18:37 WIB)
1. **Seleksi** (LSTM/GRU/TFT): latih s/d fold validasi terakhir − embargo, lalu early stopping di fold itu.
2. **Test tersegel**: latih ulang s/d awal test − embargo dengan jumlah epoch dari langkah 1, lalu prakirakan fold
   test. ARIMA-GARCH: parameter diestimasi dari data ≤ akhir training, lalu **tetap** sepanjang periode test.
3. **Produksi**: latih s/d label terakhir. Artifact ini dipakai untuk prakiraan harian.

Hasil, kesimpulan, dan versi tersimpan di `challenger_models.metrics`. Contoh kesimpulan: "top-k net −0,4% vs
baseline reversal +0,1% → perbedaan TIDAK signifikan".

### Lapisan validasi: kalibrasi dan aturan ranking (sejak schema v7)
Keduanya dipasang **hanya** dari prakiraan fold validasi (model tahap seleksi; ARIMA-GARCH: parameter diestimasi
s/d akhir training validasi). Periode test tidak dipakai untuk memilih atau menyetel apa pun.

- **Kalibrasi Platt** (`app/challengers/calibration.py`) untuk `prob_bullish`, `prob_bearish`, `prob_up`:
  `P = σ(a·logit(p) + b)`, minimal 200 baris dan dua kelas. Pemetaannya monoton, jadi AUC dan urutan tidak berubah.
  Yang membaik adalah ECE dan log loss. Test melaporkan nilai sesudah (`ece_bullish`) dan sebelum kalibrasi
  (`ece_bullish_raw`). Kalibrator ikut tersimpan di `params` dan diterapkan pada prakiraan harian. Model lama
  tanpa kalibrator tetap berjalan seperti sebelumnya.
- **Aturan ranking top-5** dipilih dari `prob_bullish`, `exp_return`, `risk_adjusted` (= return harapan / σ, tidak
  bias ke saham paling volatil), dan `prob_up`. Yang dipilih adalah aturan dengan top-k net tertinggi **di validasi**.
  Hasil test aturan itu dilaporkan sebagai `topk_by_rule_net`, dan skornya disimpan di `model_forecasts.rank_score`.
  Aturan yang dipilih pada satu run mingguan bisa berbeda dengan run berikutnya. Itu wajar dan tercatat di metrik.

### Uji signifikansi (bootstrap blok per tanggal)
Return 5 hari saling bertumpuk antar hari, sehingga bootstrap biasa akan terlalu optimistis. Karena itu dipakai
*moving-block bootstrap* dengan blok sepanjang horizon (5 hari bursa) dan 2000 ulangan pada deret harian:
- top-k net per hari (aturan terpilih): CI 95% dan p-value untuk H0 "rata-rata ≤ 0";
- IC harian: CI 95% dan p-value untuk H0 "IC ≤ 0";
- selisih **berpasangan** per hari terhadap baseline momentum dan reversal (hari yang sama, emiten yang sama);
- model aktif: prediksi test-nya tidak tersimpan per baris, jadi pembandingnya hanya kasar, yaitu apakah angka
  model aktif berada di luar CI challenger. Ini dicatat eksplisit di kesimpulan.

Kesimpulan memakai kata "lebih baik secara SIGNIFIKAN", "lebih BURUK secara signifikan", atau "perbedaan TIDAK
signifikan". Selama CI top-k net masih mencakup 0, kesimpulan selalu menyatakan model **belum layak untuk keputusan
transaksi**. Jika data test terlalu pendek (< 10 hari), CI tidak dihitung dan dilaporkan sebagai tidak tersedia.

### Prakiraan harian
Workflow **Challenger Predict** berjalan otomatis **setelah** Daily Market Pipeline sukses (`workflow_run`), jadi
tidak bergantung pada jam cron yang sering terlambat. Workflow ini memuat 400 hari bursa terakhir; test
`test_no_train_serve_skew_in_features` memastikan fiturnya identik dengan fitur dari histori penuh.

### Evaluasi live
Dashboard → Model Lab menghitung metrik dari prakiraan produksi yang horizonnya sudah lewat. Ini out-of-sample
sungguhan, tetapi butuh puluhan hari bursa sebelum bermakna.

## Kapan challenger boleh dipakai untuk keputusan

Tidak otomatis. Syarat minimal yang disarankan sebelum mempertimbangkan promosi:
1. Lebih baik dari model aktif **dan** dari baseline momentum/reversal pada top-k net return setelah biaya, di periode
   test **dan** di evaluasi live minimal 60 hari bursa. Di test, selisih terhadap baseline harus signifikan
   (bootstrap blok) dan CI 95% top-k net tidak boleh mencakup 0.
2. AUC bullish ≥ 0,52 dan ECE ≤ 0,06 (gerbang promosi model aktif).
3. Cakupan interval 80% antara 0,70 dan 0,90 (kalibrasi distribusi).
4. Konsisten di beberapa run mingguan, bukan satu run yang kebetulan bagus.

## Biaya dan kapasitas

| Sumber daya | Perkiraan |
|---|---|
| GitHub Actions (repo publik) | Train: 1 job prepare (±10 menit) + 4 job paralel (ARIMA-GARCH ±10–20 menit, LSTM/GRU ±20–35 menit, TFT ±40–80 menit; tiap fit dibatasi `max_train_minutes`). Predict: ±5–8 menit per hari, termasuk instalasi torch |
| Storage Aiven | `model_forecasts` ±900 × 4 × 252 hari ≈ 0,9 jt baris/tahun; retensi 90 hari ≈ 25 MB. Artifact: LSTM/GRU ±0,3 MB, TFT ±1 MB, ARIMA-GARCH ±0,2 MB; 2 versi terakhir disimpan |
| Bandwidth database | Train: 1× dataset penuh per minggu (±110 MB; dibagikan ke 4 job lewat artifact Actions). Predict: ±35 MB per hari. Aiven Free tidak menyebut batas egress; workflow challenger bersifat OPSIONAL (boleh di-disable bila tidak dipakai) |
| Library | torch CPU (±200 MB, dari indeks PyTorch), statsmodels, arch. Hanya di workflow challenger |

## Menjalankan

- **Pertama kali:** Actions → **Challenger Train** → Run workflow (opsional: pilih model). Lalu Actions →
  **Challenger Predict** → Run workflow. Setelah itu keduanya berjalan otomatis.
- **Lokal** (butuh `pip install torch --index-url https://download.pytorch.org/whl/cpu` dan
  `pip install -r requirements-models.txt`):
  ```bash
  python main.py challenger-prepare --out ds.npz
  python main.py challenger-train --model lstm --dataset ds.npz
  python main.py challenger-predict
  python main.py challenger-report
  ```

## Keterbatasan

- Data harga dari Yahoo dan survivorship bias (KNOWN_LIMITATIONS §1) berlaku sama untuk semua model.
- Satu periode test 6 bulan belum cukup untuk menyatakan model unggul. Lihat evaluasi live dan run berikutnya.
- ARIMA-GARCH memakai parameter tetap selama 6 bulan test. Di produksi, parameter diestimasi ulang setiap minggu.
- Model dilatih dari nol setiap minggu (tanpa warm start) agar hasil bisa direproduksi dan diaudit.
- Prakiraan 5 hari tidak memperhitungkan ARA/ARB, suspensi mendadak, atau corporate action di masa depan.
- Train/serve skew diuji: fitur 400 hari terakhir identik dengan fitur dari histori penuh, **kecuali** emiten yang
  harganya datar ±1 tahun (suspensi panjang). RSI Wilder emiten itu "membeku" pada nilai sebelum suspensi. Emiten seperti
  ini berstatus cakupan STALE/EXCLUDED, jadi prakiraannya tidak bermakna dan tidak pernah menjadi sinyal.
