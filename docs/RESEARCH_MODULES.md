# Modul riset: valuasi, foreign flow, akumulasi/distribusi, ranking terintegrasi

Dokumen ini adalah laporan upgrade (deliverable 1–17). Proyek **dikembangkan**, tidak dibuat ulang. Semua modul swing
yang sudah benar dipertahankan, dan keputusan VALUE dinilai **terpisah** dari keputusan SWING.

> Bukan nasihat investasi. Nilai wajar adalah estimasi berbasis asumsi. Sinyal akumulasi adalah bukti perilaku
> harga dan volume, bukan identitas pembeli. Sampai `evaluate-modules` dijalankan pada data pasar AS asli dengan histori
> fundamental point-in-time yang memadai, **tidak ada klaim** bahwa skor-skor ini menghasilkan return lebih baik.

---

## 1. Audit project existing (sebelum upgrade)

| Kemampuan | Status sebelum | Temuan | Status sesudah |
|---|---|---|---|
| Fundamental valuation | PARTIAL | `merge_fundamentals` (merge_asof PIT) dan `fundamental_score` sudah ada, tetapi `load_market` selalu mengirim fundamental kosong, jadi skor tidak pernah aktif di production. Belum ada tabel laporan keuangan. | `app/valuation/*`, tabel `financial_statements` (versi + first_known_date) |
| Fair value | MISSING | — | Rentang low/base/high dari 5 metode, sensitivitas DCF, `INSUFFICIENT_DATA` bila data lemah |
| Flow (AS: FINRA short volume) | MISSING | — | `app/flows/*`, tabel `foreign_flow_history` (nama dari versi BEI; saham dan nilai dipisah) |
| Accumulation | PARTIAL | Sudah ada `f_obv_slope`, `f_close_pos`, `volume_ratio`. Belum ada ADL, CMF, MFI, VWAP, divergensi, maupun tahapan. | `app/accumulation/*`, tabel `accumulation_signals` |
| Distribution | MISSING | — | `distribution_detector.py` (distribution days, churning, divergensi bearish, tekanan short FINRA) |
| Broker flow | MISSING | Data broker summary hanya tersedia dari vendor berlisensi | Antarmuka `broker_flow.py`. Nonaktif dan tidak memengaruhi skor sampai data dipasang |
| Financial quality | PARTIAL | `fundamental_score` tidak pernah mendapat data. **BUG:** filter D/E > 4 akan menolak semua bank. | `quality_score` per tipe sektor. Bug D/E diperbaiki (bank dan lembaga keuangan dikecualikan). |
| Value trap | MISSING | — | `value_trap_detector.py`: LOW/MEDIUM/HIGH/UNKNOWN beserta alasannya |
| Point-in-time | PARTIAL | Harga, berita, dan merge fundamental sudah PIT. `stocks.listed_shares` hanya snapshot. Belum ada penyimpanan fundamental PIT. | `first_known_date` / `estimated_available_date` (label `ESTIMATED_PIT`), restatement menjadi versi baru, test leakage |
| Integrated ranking | MISSING | Hanya ada ranking swing | 10 skor terpisah + 9 ranking, composite direnormalisasi (yang hilang ≠ 0) |

---

## 2. Arsitektur

```
daily pipeline (app/pipeline/jobs.py)
  ingest harga → build_dataset → scan_day (swing, + batas sektor/korelasi)
  └─ _run_research (gagal ≠ pipeline gagal; tiap tahap punya status)
       1. update_fundamentals   incremental: maks 150 emiten/run, refresh 7 hari   → financial_statements
       2. update_foreign_flow   incremental dari tanggal terakhir − 10 hari         → foreign_flow_history
       3. run_valuation         PIT pada tanggal scan                                → valuation_results
       4. run_flow_analysis     fitur 5/20/60 hari bursa + skor
       5. run_accumulation      indikator + skor + tahap (+ flow bila ada)           → accumulation_signals
       6. assemble + rank_all   10 skor terpisah, 9 ranking, VALUE decision
  → kolom riset di predictions, research_rankings_YYYYMMDD.csv, dashboard
```

| Paket | File |
|---|---|
| `app/valuation/` | `fundamental_provider`, `financial_statement_normalizer`, `valuation_metrics`, `sector_comparison`, `intrinsic_value`, `margin_of_safety`, `value_trap_detector`, `valuation_scorer`, `valuation_report` |
| `app/flows/` | `foreign_flow_provider`, `flow_normalizer`, `flow_features`, `flow_scorer`, `flow_report` |
| `app/accumulation/` | `volume_analysis`, `price_volume_analysis`, `accumulation_indicators`, `accumulation_scorer`, `distribution_detector`, `accumulation_report`, `broker_flow` |
| `app/research/` | `integrated_scoring`, `pipeline`, `report`, `evaluation`, `dashboard_data` |
| `app/risk/concentration.py` | Batas BUY per sektor dan filter korelasi (dipakai **identik** di scan dan backtest) |
| `config/research.yaml` | Semua parameter dan asumsi (berlabel). File opsional: tanpa file ini, default aman yang berlaku. |

Rumus indikator (CLV, ADL, CMF, MFI, VWAP) ditambahkan di `app/features/indicators.py` yang sudah ada, bukan
diduplikasi. Kolom akumulasi berprefiks `acc_` dan flow berprefiks `flow_`, **bukan** `f_`. Dengan begitu fitur model
ML aktif (`FEATURE_VERSION f1`) tidak berubah dan tidak perlu retrain.

---

## 3. Metode

### 3.1 Fundamental dan rasio

- **TTM** hanya dihitung bila 4 kuartal yang berakhir di akhir FY terbukti **diskret**, yaitu jumlahnya cocok dengan
  angka FY (selisih ≤ 5%). Kuartal kumulatif (YTD) tidak pernah dijumlahkan. Tanpa laporan FY, basisnya
  `INSUFFICIENT`.
- Aturan validitas:
  - PER hanya dihitung bila laba > 0; PBV hanya bila ekuitas > 0; P/FCF hanya bila FCF > 0.
  - PEG hanya bila pertumbuhan EPS 3 tahun > 2%.
  - EV/EBITDA, EV/EBIT, D/E, ROIC, dan net debt/EBITDA **tidak** dihitung untuk bank dan lembaga keuangan.
  - Laporan dalam mata uang asing tanpa kurs berarti semua multiple **tidak dihitung**. Tidak ada tebakan kurs.
- Tipe sektor (BANK / FINANCIAL / COMMODITY / PROPERTY / GENERAL / UNKNOWN) ditentukan dari nama sektor GICS
  atau Yahoo.

### 3.2 Nilai wajar (rentang)

| Metode | Isi | Tidak valid bila |
|---|---|---|
| relative | Kuartil 25/50/75 multiple peer (subsektor → sektor → tipe sektor; minimal 5 peer valid, outlier dibuang) × metrik per saham | Tidak ada kelompok ≥ 5 peer. **Tidak ada** fallback ke "seluruh pasar". |
| historical | Persentil 25/50/75 PER/PBV bulanan emiten sendiri (PIT, minimal 24 bulan, 5 tahun) × metrik terkini | Histori < 24 bulan |
| dcf | FCFF dinormalisasi (rata-rata 3 tahun; komoditas 5 tahun tanpa pertumbuhan), 5 tahun memudar ke g terminal, WACC. Grid sensitivitas ±1% WACC × ±0,5% g. Skenario low/base/high. | Bank/keuangan, FCF rata-rata ≤ 0, utang/kas tidak tersedia, nilai ekuitas ≤ 0 |
| dividend | Gordon atas DPS rata-rata 3 tahun | Dividen tidak konsisten 3 tahun, payout > 100%, laba ≤ 0 |
| justified_pb | Bank/keuangan: P/B = (ROE − g)/(COE − g) | ROE ≤ g, histori ROE < 2 tahun |

- **Kombinasi:** bobot per tipe sektor (`valuation.method_weights`) dinormalisasi ulang atas metode yang valid.
  Bila bobot valid < 40%, hasilnya `INSUFFICIENT_DATA`.
- **Confidence:**
  - HIGH: ≥ 3 metode, cakupan ≥ 70%, dan rasio dispersi ≤ 1,6;
  - MEDIUM: ≥ 2 metode dan dispersi ≤ 2,5;
  - selain itu LOW.
- **Asumsi** (risk-free 5,1% = UST 10Y, ERP 4,25%, beta 1, g 3%) adalah **ASUMSI**, bukan data pasar. Health check memberi
  WARN bila `assumptions_as_of` > 180 hari.

### 3.3 Margin of safety dan status

- `MoS = (FV − harga) / FV`, dihitung untuk base dan versi konservatif (FV low).
- Status valuasi:
  - `DEEP_VALUE`: MoS ≥ 40%, MoS konservatif ≥ 0, dan confidence bukan LOW;
  - `UNDERVALUED`: MoS ≥ 20%;
  - `OVERVALUED`: MoS ≤ −15%;
  - `FAIRLY_VALUED`: di antara keduanya;
  - `INSUFFICIENT_DATA`.

### 3.4 Value trap

Pemeriksaan bernilai True, False, atau **None** (tidak dapat dinilai):

- pendapatan atau laba turun 2 tahun;
- rugi;
- margin menyusut;
- ekuitas negatif;
- OCF/laba < 0,5;
- leverage naik / D/E > 2;
- coverage < 2;
- FCF negatif berulang;
- dividen dipotong;
- falling knife (di bawah MA200 dan turun > 40% dari puncak);
- data basi;
- kualitas data `CHECK`.

Penilaian risiko:

- `HIGH`: poin ≥ 4 atau ada tanda berat.
- `MEDIUM`: 2–3 poin.
- `LOW`: ≤ 1 poin.
- `UNKNOWN`: bila < 4 pemeriksaan dapat dinilai. UNKNOWN **bukan** berarti aman.

### 3.5 Flow (versi AS: FINRA daily short sale volume)

- Pasar AS tidak mempublikasikan foreign flow. Slot modul ini diisi data **FINRA Reg SHO daily short sale volume**
  (gratis, seluruh saham AS, terbit ≤ 18:00 ET hari yang sama). Pemetaan: `foreign_sell_shares` = ShortVolume,
  `foreign_buy_shares` = TotalVolume − ShortVolume, `total_volume_shares` = TotalVolume (volume yang dilaporkan ke
  FINRA, bukan volume konsolidasi). Nama kolom/tabel dipertahankan agar skor dan ranking tidak berubah.
- Rasio short FINRA struktural ±50% (aktivitas market maker), jadi fitur memakai **simpangan dari baseline 120 hari
  bursa saham itu sendiri** (`foreign_flow.baseline_days`; baseline hari D hanya memakai data ≤ D−1). NET_BUYING =
  tekanan short di bawah kebiasaan; NET_SELLING = di atasnya.
- Nilai dolar = saham × VWAP harian, berlabel **`ESTIMATED_VALUE`**; estimasi tidak pernah disimpan sebagai data mentah.
- Jendela 5/20/60 dihitung dalam hari bursa saham. Hari tanpa data **bukan nol**; hari itu mengurangi cakupan.
- Skor 0–100 dan status:
  - `FOREIGN_FLOW_UNAVAILABLE`: tidak ada data sama sekali;
  - `INSUFFICIENT_DATA`: cakupan 20 hari < 80%;
  - selain itu `STRONG_NET_BUYING`, `NET_BUYING`, `NEUTRAL`, `NET_SELLING`, atau `STRONG_NET_SELLING`.

### 3.6 Akumulasi dan distribusi

- **Indikator:**
  - OBV, ADL, dan trend 20/60 hari (dinormalisasi Σ volume);
  - CMF20, MFI14, CLV;
  - RVOL (pembagi = rata-rata 20 hari **sebelumnya**);
  - VWAP20 (dari nilai transaksi bila ada);
  - rasio volume naik/turun, volume dry-up;
  - distribution/accumulation days, range base, breakout, divergensi, churning.
- **Status:**
  - `STRONG_ACCUMULATION_SIGNAL` **hanya** bila skor ≥ 75, CMF > 0,05, OBV naik, **dan** foreign flow tersedia
    dengan skor ≥ 60. Tanpa flow, status maksimum adalah `MODERATE_*`, dan pembatasan ini dicatat sebagai bukti.
  - Status distribusi berlaku simetris.
  - Histori < 60 hari bursa menghasilkan `INSUFFICIENT_DATA`.
- **Tahap:** `EARLY_ACCUMULATION_WATCHLIST`, `BREAKOUT_CONFIRMED`, `DISTRIBUTION_WARNING`, `NO_CLEAR_SIGNAL`.
- **Confidence:**
  - LOW bila likuiditas rendah;
  - HIGH bila flow tersedia;
  - MEDIUM bila tidak.

### 3.7 Skor terpisah dan 9 ranking

- **Skor terpisah:**
  - `value_score`, `business_quality_score`, `value_trap_risk`;
  - `foreign_flow_score`;
  - `accumulation_score`, `distribution_risk_score`;
  - `technical_setup_score`, `ml_probability_score`;
  - `liquidity_score`, `market_regime_score`.
- **Ranking:** `best_value`, `quality_value`, `accumulation`, `foreign_buying`, `swing_setup`,
  `value_accumulation`, `value_swing`, `momentum_flow`, `integrated`. Bobotnya ada di `research_scoring.rankings`.
- **Composite** = rata-rata berbobot skor yang tersedia. Bila cakupan bobot < 60%, emiten **tidak diranking** (tidak
  diberi angka rendah). Ranking berbasis nilai mengecualikan value trap HIGH.
- **VALUE decision:** `VALUE_CANDIDATE`, `VALUE_WATCH`, `VALUE_TRAP_RISK`, `NOT_UNDERVALUED`, `INSUFFICIENT_DATA`.
- **SWING decision** (BUY/WATCHLIST/WAIT/AVOID) **tidak diubah** oleh modul valuasi.

---

## 4. Point-in-time

| Data | Kapan boleh dipakai |
|---|---|
| Harga, flow hari D | Setelah penutupan D (daily 23:15 UTC = 19:15 ET) |
| Laporan dari sumber PIT (CSV dengan `publication_date` asli, feed berlisensi) | `first_known_date = publication_date` |
| Laporan tanpa tanggal publikasi (snapshot Yahoo) | `first_known_date` = tanggal sistem pertama kali mengambilnya. **Tidak** dipakai mundur di backtest. |
| Restatement (angka periode yang sama berubah) | Versi baru dengan `first_known_date` = hari diketahui. Versi lama **tetap ada**, sehingga backtest melihat angka lama. |
| Mode riset `--estimated-pit` | `estimated_available_date` = akhir periode + 150 hari (FY) / 100 hari (Q). Semua hasil berlabel **`ESTIMATED_PIT`** dan tidak disimpan ke `valuation_results`. |

Test yang membuktikan PIT:

- **`test_valuation.py`:** belum dipublikasi berarti tidak terlihat; restatement memakai versi lama sebelum tanggal
  diketahui; snapshot tidak dipakai mundur; valuasi Maret 2024 hanya memakai FY2022.
- **`test_flows_accumulation.py`:** menambah data masa depan tidak mengubah fitur flow maupun indikator akumulasi;
  satuan saham; tidak ada duplikat.

---

## 5. Database (additive, histori tidak dihapus)

`SCHEMA_VERSION` 3. Migrasi otomatis lewat `migrate()`: CREATE IF NOT EXISTS + ADD COLUMN, tanpa DROP.

| Tabel baru | Kunci UPSERT | Isi |
|---|---|---|
| `financial_statements` | stock_id, period_end, period_type, source, version | items JSON kanonik, items_hash, publication_date, first_known_date, estimated_available_date, retrieved_at, source, quality_status/notes |
| `valuation_results` | stock_id, as_of_date | Rentang FV, MoS, status, confidence, skor, value trap, metode/metrik (JSON ringkas ±2,4 KB) |
| `foreign_flow_history` | stock_id, date, market_segment | FINRA short/non-short/total volume, value_type, units, source, source_timestamp, quality |
| `accumulation_signals` | stock_id, date | Hanya sinyal non-netral (default), komponen dan bukti |

- **Kolom baru:** 11 kolom riset di `predictions`; `category/status/detail/checked_at` di `data_sources`;
  `fundamentals_checked_at` di `stocks`.
- **Sengaja tidak dibuat** (menghindari duplikasi):
  - `fundamental_metrics`: diturunkan dari laporan;
  - `fair_value_estimates`: masuk ke `methods` JSON;
  - tabel evaluasi valuasi: memakai `backtest_runs` kind `module_evaluation`;
  - tabel status sumber riset: `data_sources` diperluas.

**Retensi** (`db-maintenance`, otomatis saat database ≥ 85% kuota):

- `valuation_results` harian > 14 hari dihapus, **kecuali valuasi akhir bulan**;
- `accumulation_signals` > 365 hari dihapus;
- `foreign_flow_history` > 7 tahun dihapus;
- `financial_statements` **tidak pernah** dihapus.

**Perkiraan pertumbuhan** (±920 emiten):

- foreign flow ±45 MB/tahun (bila file tersedia setiap hari);
- valuasi ±22 MB/tahun (akhir bulan) + ±30 MB (14 hari bergulir);
- laporan keuangan ±5 MB.

---

## 6. Sumber data dan kredensial

| Kebutuhan | Tanpa kredensial (sekarang) | Untuk kualitas penuh |
|---|---|---|
| Laporan keuangan | **SEC EDGAR** companyfacts (10-K/10-Q, point-in-time dari tanggal `filed`; butuh Variable `SEC_USER_AGENT`). Cadangan: snapshot Yahoo. | CSV vendor berlisensi (`fundamentals-template`) bila butuh emiten 20-F/40-F |
| Flow | **FINRA** daily short sale volume (otomatis, gratis) | API vendor beli/jual sungguhan (`FOREIGN_FLOW_API_URL`, `FOREIGN_FLOW_API_KEY`) + `baseline_days: null` |
| Kurs laporan non-USD | `data/raw/fx/fx_rates.csv` (date,currency,rate) | — |
| Broker summary | Tidak ada padanan publik di AS; skor tidak terpengaruh | — |

Semua kredensial hanya lewat GitHub Secrets/Variables: tidak ada di repo, CSV, report, maupun Docker image.

---

## 7. Dashboard (4 halaman baru)

| Halaman | Isi |
|---|---|
| **Undervalued Screener** | Filter MoS, status, value trap, dan quality. Menampilkan rentang FV, confidence, alasan value trap, tanggal fundamental diketahui, serta status sumber data. |
| **Short Flow (FINRA)** | Bila tidak ada data: `FOREIGN_FLOW_UNAVAILABLE` + status sumber. Bila ada: rasio short volume seluruh universe, tabel per saham, dan detail rasio short per ticker. |
| **Accumulation / Distribution** | Filter tahap, skor, risiko distribusi, dan bukti per emiten. |
| **Integrated Research** | 9 ranking, tabulasi silang VALUE × SWING, dan STOCK RESEARCH REPORT per emiten (dihitung ulang secara PIT). |

Logika data ada di `app/research/dashboard_data.py` dan dites tanpa Streamlit.

---

## 8. Perintah baru

| Perintah | Fungsi |
|---|---|
| `python main.py research AAPL [--date]` | STOCK RESEARCH REPORT (PIT) |
| `python main.py rankings --ranking best_value --top 20` | Salah satu dari 9 ranking |
| `python main.py valuation [--date] [--estimated-pit]` | Valuasi seluruh universe |
| `python main.py fundamentals-update [--tickers AAPL,MSFT]` | Ambil fundamental (incremental) |
| `python main.py fundamentals-template` | Template CSV fundamental kanonik |
| `python main.py foreign-flow-update [--full]` | Muat file/API foreign flow |
| `python main.py evaluate-modules [--estimated-pit] [--json]` | Evaluasi historis 9 varian + IC faktor (juga dijalankan di Weekly Backtest) |

---

## 9. Evaluasi historis (`evaluate-modules`)

- **Baseline:**
  - Pipeline sinyal **identik** dengan scan, tetapi dengan prediksi **netral** (teknikal saja).
  - Model ML aktif tidak dipakai untuk periode lampau, karena model itu dilatih dengan data hingga baru-baru ini
    (kebocoran). Kontribusi ML dievaluasi oleh walk-forward backtest yang melatih ulang per fold.
- **9 varian:**
  - V1 baseline;
  - V2 tanpa distribusi;
  - V3 + akumulasi;
  - V4 + foreign flow;
  - V5 tanpa value trap;
  - V6 + value;
  - V7 + quality;
  - V8 ranking integrated;
  - V9 value + akumulasi.
- **Pengaturan evaluasi:**
  - Ambang berasal dari config **sebelum** evaluasi.
  - Periode VALIDATION dan TEST dilaporkan terpisah, dan test tidak dipakai untuk memilih.
  - Varian dengan cakupan data < 50% sinyal menghasilkan `INSUFFICIENT_DATA`, bukan angka.
- **Breakdown** per regime, sektor, setup, status valuasi, dan status akumulasi (jumlah trade, win rate, rata-rata
  return bersih).
- **IC faktor:**
  - Spearman per tanggal: value/quality terhadap return 60/120/250 hari (akhir bulan, PIT); akumulasi/flow terhadap
    return 5/10/20 hari.
  - Kurang dari 12 tanggal menghasilkan `INSUFFICIENT_HISTORY`. t-stat untuk tanggal yang bertumpuk diberi catatan
    "optimistis".
- **Pembanding momentum:** IC momentum 20 hari ikut dilaporkan, beserta korelasi rank skor akumulasi terhadap
  momentum. Skor akumulasi sebagian dibangun dari perilaku harga, jadi IC-nya bisa sekadar mengulang momentum.
  Korelasi > 0,7 berarti informasi barunya kecil.
- **Data sintetis:** hasil dari `make-sample` hanya membuktikan alurnya bekerja, dan **tidak bermakna** secara
  finansial. Contohnya, pada data contoh IC akumulasi (+0,13 / 5 hari) hampir sama dengan IC momentum murni
  (+0,125). Ini sifat generator data sintetis, bukan bukti kemampuan prediksi.

---

## 10. Self-audit (P0 = kritis … P3 = minor)

| # | Prioritas | Temuan | Tindakan | Status |
|---|---|---|---|---|
| 1 | P0 | Filter fundamental D/E > 4 menolak semua bank (D/E bank normal 5–8×) | Dibuat sektor-aware + test | **Diperbaiki** |
| 2 | P0 | Risiko data karangan: foreign flow kosong dianggap nol, skor hilang dianggap 0 | Semua jalur memakai None + label (`FOREIGN_FLOW_UNAVAILABLE`, `INSUFFICIENT_DATA`), composite direnormalisasi, ada test | **Diperbaiki** |
| 3 | P0 | Fundamental tanpa tanggal publikasi bisa bocor ke backtest | `first_known_date` + versi restatement; mode estimasi diberi label dan tidak disimpan | **Diperbaiki** |
| 4 | P1 | `technical_score` mengisi komponen hilang dengan 0 (emiten tanpa data sektor/regime dihukum sistematis) | Renormalisasi bobot atas komponen tersedia; cakupan < 70% → NaN (tidak BUY) | **Diperbaiki** |
| 5 | P1 | Tidak ada batas konsentrasi: BUY bisa menumpuk di satu sektor atau saham yang sangat berkorelasi | `app/risk/concentration.py` (maks 2 per sektor, korelasi 60 hari > 0,8 → WATCHLIST), identik di scan dan backtest | **Diimplementasikan** |
| 6 | P1 | Tidak ada monitoring kalibrasi prediksi live | `live_calibration` (ECE/Brier, hit rate BUY per regime) + health `LIVE CALIBRATION` | **Diimplementasikan** |
| 7 | P1 | Data basi tidak terlihat | Health: `FUNDAMENTALS`, `FOREIGN FLOW` (basi > 7 hari tidak dipakai), `VALUATION ASSUMPTIONS` (> 180 hari); value trap memeriksa laporan > 450 hari | **Diimplementasikan** |
| 8 | P1 | Kuota database paket gratis: tabel riset bisa membengkak (valuasi JSON ±4 KB/emiten/hari) | JSON ringkas (±2,4 KB), retensi harian 14 hari + akhir bulan, retensi akumulasi dan flow | **Diimplementasikan** |
| 9 | P1 | Laporan non-USD dibandingkan langsung dengan harga USD | Kurs wajib; tanpa kurs → `INSUFFICIENT_DATA`; quality `CHECK` | **Diimplementasikan** |
| 10 | P2 | `fundamental_filter` swing hanya aktif bila ada kolom `fund_*` yang tidak pernah dikirim `load_market` | Sengaja **tidak** disambungkan. Bila disambungkan, filter akan aktif di live tetapi tidak di backtest (tidak ada histori PIT), sehingga hasil backtest menyesatkan. Keputusan VALUE terpisah. | Terbuka (by design) |
| 11 | P2 | Drift fitur model (PSI) belum dimonitor | Butuh statistik distribusi training yang disimpan bersama model; direncanakan bersama bump `FEATURE_VERSION` berikutnya | Terbuka |
| 12 | P2 | Spread bid-ask, halt LULD, dan jam pra/pasca-pasar belum dimodelkan di backtest | Slippage 5 bps + eksekusi konservatif t+1; lihat KNOWN_LIMITATIONS | Terbuka |
| 13 | P2 | Dashboard publik bisa menampilkan data ke siapa pun | Kredensial di server (aman), tetapi isi bisa dilihat siapa pun. Gunakan app privat / viewer auth Streamlit. | Dokumentasi |
| 14 | P2 | Short volume FINRA hanya mencakup volume yang dilaporkan ke FINRA (off-exchange), bukan seluruh pasar | Dipakai sebagai simpangan relatif terhadap baseline saham sendiri; status UNAVAILABLE/STALE terlihat di health dan dashboard | Diketahui |
| 15 | P3 | Dividend model memakai dividen kas dibayar (arus kas), bukan DPS yang diumumkan | Rata-rata 3 tahun + syarat konsistensi | Diketahui |
| 16 | P3 | Beta default 1,0 untuk semua emiten | Asumsi berlabel; sensitivitas DCF menutup sebagian | Diketahui |
| 17 | P3 | Kelompok peer bergantung pada taksonomi sektor Yahoo bila `universe.sectors_file` kosong | Universe indeks sudah membawa sektor GICS; hanya saham dari file tanpa sektor yang memakai Yahoo | Dokumentasi |
| 18 | P3 | Historical multiple memakai jumlah saham terkini (harga sudah split-adjusted); rights issue membuat bias | Dicatat di docstring | Diketahui |

---

## 11. Cara verifikasi

```bash
# test lengkap (SQLite; + PostgreSQL bila TEST_POSTGRES_URL diset — sama seperti workflow Tests)
python -m pytest -q
# setelah deploy: jalankan Daily, lalu
python main.py health --no-provider      # FUNDAMENTALS / FOREIGN FLOW / VALUATION ASSUMPTIONS / LIVE CALIBRATION
python main.py research AAPL             # laporan emiten, cek label UNAVAILABLE/INSUFFICIENT_DATA bila data kosong
python main.py rankings --ranking best_value
python main.py evaluate-modules          # IC & varian; INSUFFICIENT_HISTORY wajar sampai histori PIT terkumpul
```

Di Supabase SQL Editor:

```sql
SELECT name, category, status, detail, checked_at FROM data_sources WHERE category IN ('fundamental','foreign_flow');
SELECT valuation_status, COUNT(*) FROM valuation_results
 WHERE as_of_date = (SELECT MAX(as_of_date) FROM valuation_results) GROUP BY 1;
SELECT accumulation_status, COUNT(*) FROM predictions
 WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions) GROUP BY 1;
```

---

## 12. Ringkasan deliverable

1. **Audit:** §1.
2. **Arsitektur dan daftar file baru:** §2.
3. **File yang diubah:**
   - `schema.py`, `repository.py`, `maintenance.py`;
   - `jobs.py`, `health.py`, `evaluate.py`;
   - `scanner.py`, `backtest/runner.py`, `scoring.py`;
   - `fundamental_sentiment.py`, `indicators.py`;
   - `config.py`, `main.py`, `dashboard/app.py`;
   - workflow `daily` dan `weekly_backtest`;
   - `.env.example`, README.
4. **Migrasi database:** §5.
5. **Provider dan sumber data:** §6.
6. **Kredensial yang dibutuhkan:** §6. Tidak ada yang wajib; semuanya opsional.
7. **Metode valuasi dan fair value:** §3.1–3.3.
8. **Value trap:** §3.4.
9. **Foreign flow:** §3.5.
10. **Akumulasi dan distribusi:** §3.6.
11. **Skor terintegrasi dan ranking:** §3.7.
12. **Point-in-time:** §4.
13. **Daily pipeline:** §2.
14. **Dashboard:** §7.
15. **Laporan emiten:** `python main.py research TICKER` (contoh di README).
16. **Self-audit + implementasi:** §10.
17. **Evaluasi dan cara verifikasi:** §9, §11.
