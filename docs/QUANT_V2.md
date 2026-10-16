# Quant v2 — audit dan perbaikan (16 Okt 2026)

Dokumen ini mencatat audit sistem produksi (dashboard Streamlit, data Aiven) dari sudut pandang quant, kelemahan yang
ditemukan, dan apa yang diubah di kode. Tujuannya bukan "menyamai Renaissance" — itu tidak mungkin dengan data harian
gratis, tanpa data tick/order book, tanpa biaya eksekusi institusional, dan tanpa ribuan sinyal yang diteliti puluhan
tahun. Yang bisa ditiru adalah **prosesnya**: setiap sinyal harus dibuktikan secara statistik, banyak sinyal lemah
digabung, risiko pasar dinetralkan, dan biaya selalu dihitung.

---

## 1. Temuan audit (data produksi)

| # | Temuan | Bukti | Dampak |
|---|---|---|---|
| 1 | **Label salah sasaran.** Kelas "bullish" = return mentah 5 hari > +3% | `labels.mode: fixed` | Model belajar "saham volatil/beta tinggi di pasar naik", bukan "saham yang akan mengungguli". AUC 0,62 terlihat bagus tetapi sebagian besar adalah beta + volatilitas |
| 2 | **Hasil backtest belum signifikan.** Strategi +3,8%, Sharpe 0,47 dalam 6 bulan | standard error Sharpe untuk 6 bulan ≈ 1,4 | Sharpe 0,47 ± 1,4 → tidak bisa dibedakan dari nol |
| 3 | **Benchmark tidak adil.** Strategi hanya 43% terinvestasi, tetapi dibandingkan dengan S&P 500 100% | exposure 43% | Kalah dari buy & hold belum tentu buruk; menang juga belum tentu alpha. Pembanding yang benar = beta × S&P 500 |
| 4 | **52% exit kena stop**, sebagian karena gap earnings | distribusi exit backtest | Risiko event yang tidak dimodelkan sama sekali |
| 5 | **Biaya 2,7% modal** dalam 6 bulan | total fee backtest | Sinyal dengan ekspektasi < biaya pulang-pergi tetap dieksekusi |
| 6 | **Tidak ada normalisasi cross-sectional.** Fitur skala mentah per saham | `app/features/technical.py` | Distribusi fitur bergeser antar regime → model tidak stabil |
| 7 | **Anomali harga yang terdokumentasi tidak dipakai** | — | Momentum 12-1, pembalikan residual jangka pendek, jarak ke high 52 minggu, efek MAX, volatilitas idiosinkratik, PEAD |
| 8 | **FINRA 0 baris tanpa error** | Data Health | HTTP 403 dianggap "belum terbit" → fitur short-flow diam-diam kosong |
| 9 | **S&P 400 tidak terambil** (universe 503, bukan ±900) | Data Health | Wikipedia menolak User-Agent tanpa kontak dari IP datacenter |

## 2. Yang diubah

### 2.1 Label: return residual cross-sectional (`labels.mode: xs_resid`)
`app/models/labels.py`
- `y_xs_5d = return saham 5H − beta60 × return S&P 500 5H` (beta dihitung hanya dari data ≤ t).
- Dibagi `vol20 × √5` → skor-z residual, lalu **diranking per tanggal**: sepertiga teratas = kelas 2, sepertiga
  terbawah = kelas 0.
- Hasilnya: tiap hari selalu ada ±1/3 kelas bullish, apa pun arah pasar; saham beta tinggi tidak lagi mendominasi.
- Mode lama (`fixed`, `vol_scaled`) masih tersedia di `config/model.yaml`.
- AUC turun ke ±0,55–0,59 (sampel). Ini **lebih jujur**: di saham likuid AS, AUC 0,55 yang stabil pada target
  market-neutral jauh lebih berharga daripada 0,62 pada target yang bocor beta.

### 2.2 Fitur alpha (`FEATURE_VERSION = f2`)
`app/features/alpha.py`, `app/features/earnings.py`

| Fitur | Riset acuan |
|---|---|
| `f_mom_12_1`, `f_mom_6_1` | Jegadeesh & Titman (1993) |
| `f_resid_ret5`, `f_resid_ret21` (pembalikan residual) | Jegadeesh (1990); Blitz dkk. (2013) |
| `f_dist_52w_high` | George & Hwang (2004) |
| `f_max_ret21`, `f_min_ret21`, `f_skew60` | Bali, Cakici & Whitelaw (2011) |
| `f_idio_vol60`, `f_beta60` | Ang dkk. (2006) |
| `f_ear3` — abnormal return 3 hari di sekitar rilis laba | PEAD: Bernard & Thomas (1989) |
| `f_days_since_earn`, `f_days_to_earn_est` | siklus pengumuman laba |
| `f_xs_*` — rank 0..1 per tanggal untuk 19 fitur | normalisasi cross-sectional |

Semua jendela hanya ke belakang. Rank hanya di antara saham yang bisa ditransaksikan pada tanggal yang sama.
`f_ear3` baru tersedia H+1 setelah hari reaksi selesai (tidak ada look-ahead).

### 2.3 Tanggal rilis laba dari SEC (gratis, point-in-time)
- Sumber: `data.sec.gov/submissions` — 8-K dengan **Item 2.02** (Results of Operations), tanpa amandemen.
- Hari reaksi: diterima ≥ 16:00 ET → hari bursa berikutnya; pagi/siang → hari itu.
- Tabel baru `earnings_events` (skema v8). Ukuran ±25 ribu baris untuk 900 saham × 7 tahun (< 5 MB).
- Pembaruan:
  - **backfill** otomatis sebelum training bila tabel masih kosong (setup/upgrade), atau manual
    `fundamentals-update --earnings-backfill` (sudah dipakai workflow Fundamental Refresh);
  - **pantauan harian**: saham yang perkiraan rilisnya jatuh minggu ini dicek setiap Daily
    (`fundamentals.earnings_watch_max`, default 300), sehingga event baru terdeteksi H+1 seperti asumsi backtest.

### 2.4 Filter NO-TRADE baru
`app/strategy/filters.py`, `config/trading.yaml`
- `earnings_within_holding` — perkiraan rilis laba ≤ 16 hari kalender ke depan (horizon 5 hari bursa + buffer)
  → tidak masuk. Menghindari gap yang menyebabkan banyak stop-out. Tanpa data earnings → tidak memblokir.
- `expected_return_below_cost` — ekspektasi return bersih harus > `cost_multiple × biaya pulang-pergi`
  (default 1× ≈ 0,31%).

### 2.5 Statistik signifikansi di setiap backtest
`app/backtest/stats.py` — ditampilkan di CLI `backtest`, ringkasan workflow, dan dashboard → Backtest.
- **Alpha & beta** (OLS return harian strategi vs S&P 500), t-stat alpha, information ratio.
- **Benchmark adil**: return beta × S&P 500 pada periode yang sama.
- **Selang kepercayaan 90% Sharpe** (stationary block bootstrap, blok rata-rata 5 hari) + peluang Sharpe ≤ 0.
- **Probabilistic Sharpe Ratio** dan **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) — memotong bukti
  sesuai jumlah konfigurasi yang dicoba (`n_trials` = kandidat model × grid tuning).
- Verdict otomatis: `BUKTI KUAT` (DSR ≥ 0,95 dan t-alpha ≥ 2) · `BUKTI LEMAH` (DSR ≥ 0,80 atau t-alpha ≥ 1,5) · `BELUM ADA BUKTI EDGE`.

### 2.6 Riset faktor (`python main.py factor-report`)
`app/research/factor_report.py` — dijalankan otomatis di Weekly Backtest, tampil di dashboard → Model Performance.
- Rank IC harian setiap fitur vs return residual 5 hari, hanya tanggal yang tidak tumpang tindih (tiap 5 hari).
- Seluruh histori vs 12 bulan terakhir (apakah sinyal masih hidup?).
- Panduan baca: |t| ≥ 2 signifikan; IC 0,02–0,05 sudah berguna di saham likuid AS.

### 2.7 Evaluasi model
- **Rank IC** (`rank_ic_mean`, IR, t-stat, hit rate) per fold, tercatat di registry model.
- Model lama (f1) tidak kompatibel → Daily otomatis melatih ulang sekali. Bila gerbang gagal dan model aktif tidak
  kompatibel, model baru dipromosikan sebagai BASELINE (dilaporkan `MODEL STATUS: WARN`).

### 2.8 Sumber data yang tidak lagi gagal diam-diam
- FINRA: header mirip browser; 403 berulang tanpa satu pun sukses → error jelas di log dan Data Health.
- Wikipedia: User-Agent memuat email dari `SEC_USER_AGENT` sesuai kebijakan Wikimedia, plus cadangan REST API.
  Status per indeks (`universe_sp500`, `universe_sp400`) tampil di Data Health.

## 3. Cara membaca hasil setelah upgrade

1. **Weekly Backtest → verdict.** Jangan menambah modal sebelum verdict test period minimal `BUKTI LEMAH` dua
   minggu berturut-turut, dan idealnya `BUKTI KUAT`.
2. **Alpha, bukan return.** Return strategi harus mengalahkan *beta-matched benchmark*, bukan S&P 500 penuh.
3. **Factor IC.** Fitur dengan |t| < 1 di 12 bulan terakhir bisa dihapus dari model. Fitur dengan IC yang berbalik
   tanda antara histori penuh dan 12 bulan adalah tanda sinyal sudah mati / ter-arbitrase.
4. **Ekspektasi realistis.** Sistem swing harian dengan data gratis yang bagus menghasilkan IC ±0,02–0,04 dan
   Sharpe bersih 0,5–1,0 dalam jangka panjang — dengan drawdown yang tetap nyata. Klaim di atas itu harus dicurigai.

## 4. Yang BELUM bisa (dan kenapa)

- Data intraday, order book, opsi, dan estimasi analis (surprise EPS) tidak tersedia gratis dengan kualitas cukup.
  `f_ear3` adalah proksi surprise dari reaksi harga, bukan surprise EPS sebenarnya.
- Portofolio long-short (netral pasar penuh) tidak dibangun: akun ritel di Indonesia umumnya tidak bisa short saham
  AS dengan biaya wajar. Label residual tetap dipakai untuk memilih saham yang *relatif* lebih kuat.
- Universe tetap ±900 saham (batas 1 GB Aiven Free). Lihat DATABASE_DECISION.md untuk opsi perluasan.
