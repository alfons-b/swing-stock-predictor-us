# swing-stock-predictor-us — AI Swing Stock Predictor untuk Saham AS (cloud-first)

**Analytical decision support system** untuk swing trading saham Amerika Serikat (NYSE/Nasdaq, universe S&P 500 +
S&P MidCap 400). Fiturnya sama persis dengan versi BEI (`swing-stock-predictor`): probabilitas, level
entry/SL/TP, risk/reward, ukuran posisi, valuasi, flow, akumulasi/distribusi, 9 ranking riset, backtest walk-forward,
model challenger, dan dashboard. Sistem ini bukan sistem profit terjamin dan berani menjawab **NO TRADE**.

Seluruh pipeline berjalan di **GitHub Actions** dan menyimpan data ke **Aiven for PostgreSQL (Free)** — database
terpisah dari Neon (versi BEI) dan Supabase (aplikasi lain), yang **diblokir** di produksi. Laptop tidak perlu hidup.

```
GitHub Actions (cron UTC / Run workflow) ─► python main.py daily ─► Aiven PostgreSQL ◄─ Dashboard (Streamlit Cloud)
        │                                   │
        │                                   ├─ universe S&P 500+400 (Wikipedia) · harga Yahoo (^GSPC) · SEC EDGAR · FINRA
        │                                   ├─ fitur → regime → sektor → setup → ML → risk → ranking
        │                                   └─ prediksi, sinyal, report, log, status run → database
        └─ job hidup hanya selama run: START → PROCESS → SAVE → EXIT
```

**➡️ Panduan setup lengkap langkah demi langkah: [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md).**
**➡️ Audit quant & perbaikan v2 (label residual, fitur alpha, earnings SEC, uji signifikansi): [`docs/QUANT_V2.md`](docs/QUANT_V2.md).**

---

## Apa yang berbeda dari versi BEI

| Aspek | Versi BEI | Versi US (repo ini) |
|---|---|---|
| Universe | File Daftar Saham BEI (unduh manual) | Konstituen S&P 500 + S&P 400 dari Wikipedia, otomatis, dengan sektor GICS |
| Indeks acuan | IHSG (`^JKSE`) | S&P 500 (`^GSPC`) |
| Ticker | `BBCA` (Yahoo `BBCA.JK`) | `AAPL`, `BRK-B` (format Yahoo) |
| Kalender | Asia/Jakarta, libur BEI diisi manual | America/New_York, libur NYSE 2018–2027 sudah terisi |
| Aturan order | fraksi harga BEI, lot 100, ARA/ARB | tick $0,01, saham utuh (tanpa lot), tanpa batas harian |
| Biaya backtest | 0,15% / 0,25%, slippage 10 bps | 0,10% / 0,11%, slippage 5 bps |
| Mata uang | Rupiah, modal default Rp100 juta | USD, modal default $10.000 |
| Fundamental | snapshot Yahoo (bukan point-in-time) | **SEC EDGAR 10-K/10-Q point-in-time** (gratis) → cadangan Yahoo |
| Flow | Foreign flow dari file Ringkasan Saham (manual) | **FINRA daily short volume** (gratis, otomatis) |
| Database | Neon | Aiven (Neon & Supabase ditolak di produksi) |
| Jadwal Daily | 18:30 WIB | 23:15 UTC = 06:15 WIB (setelah bursa AS tutup) |

## Ringkasan setup (detail di SETUP_GUIDE)

1. Buat repo GitHub baru (disarankan **publik** agar menit Actions tidak terbatas), push isi folder ini.
2. Buat service **Aiven for PostgreSQL → Free**, salin Service URI.
3. GitHub → Settings → Secrets and variables → Actions:
   - Secret `DATABASE_URL` = Service URI Aiven
   - Variable `SEC_USER_AGENT` = `Nama Anda email@domain.com`
   - Variable `PORTFOLIO_VALUE` = modal Anda dalam USD (opsional, default `10000`)
4. Actions → **Tests** → **Initial Setup** → **Database Health Check** → **Daily Market Pipeline** (Run workflow).
5. `python scripts/create_dashboard_role.py` (role read-only), lalu deploy `dashboard/app.py` di Streamlit Community Cloud.

## Perintah

| Perintah | Fungsi |
|---|---|
| `python main.py health [--json] [--no-provider] [--strict]` | DATABASE, PROVIDER, UNIVERSE, tanggal pasar vs database, model, pipeline, freshness |
| `python main.py setup` | Inisialisasi lengkap. Idempoten |
| `python main.py daily [--skip-ingest]` | Pipeline harian penuh |
| `python main.py update-data [--actions]` | Unduh **hanya** data yang hilang, validasi, UPSERT |
| `python main.py scan` / `predict AAPL` | Scan seluruh universe / analisis satu saham |
| `python main.py research AAPL` | STOCK RESEARCH REPORT: valuasi SEC, short flow FINRA, akumulasi, skor & ranking |
| `python main.py rankings --ranking best_value` | Salah satu dari 9 ranking riset |
| `python main.py fundamentals-update [--tickers AAPL,MSFT] [--earnings-backfill]` / `foreign-flow-update` / `valuation` | Modul riset terpisah (laporan SEC + tanggal rilis laba 8-K) |
| `python main.py train [--if-due]` · `backtest [--folds N]` · `evaluate` · `evaluate-modules` | Model & evaluasi (backtest mencetak alpha, beta, CI Sharpe, Deflated Sharpe, verdict) |
| `python main.py factor-report [--top N]` | Rank IC tiap fitur vs return residual 5 hari (bukti per sinyal) |
| `python main.py models [--promote model_v00X]` | Daftar versi model / manual override |
| `python main.py db-maintenance [--dry-run] [--full]` · `db-schema` | Retensi + VACUUM · tulis `sql/schema_postgres.sql` |
| `python main.py challenger-train --model lstm\|gru\|tft\|arima_garch` | Model challenger (butuh `requirements-models.txt`) |
| `--as-of YYYY-MM-DD` (opsi global, sebelum nama perintah) | Simulasi tanggal (test/backfill) |

Exit code: 0 sukses · 1 gagal (Actions merah) · 2 config salah · 3 health kritis gagal.

## Bagaimana sistem menjaga integritas

- **Incremental & idempoten.** Tiap saham hanya mengunduh tanggal yang hilang (dihitung dengan jam New York);
  UPSERT dengan `UNIQUE(stock_id, date)`; run yang terlewat dikejar otomatis.
- **Split.** Yahoo menyesuaikan split retroaktif; bila close baru berbeda > 20% dari close terakhir di database,
  histori ticker itu diunduh ulang.
- **Survivorship.** Saham yang keluar dari indeks ditandai nonaktif, tidak dihapus. Daftar dari sumber cadangan
  (Wikipedia gagal) **tidak pernah** menonaktifkan saham.
- **Point-in-time.** Laporan keuangan baru dipakai sejak tanggal `filed` di SEC; restatement tahun berikutnya tidak
  bocor ke masa lalu. Baseline flow hari D hanya memakai data ≤ D−1.
- **Tanpa look-ahead.** `tests/test_no_data_leakage.py` membuktikan fitur dari data penuh = fitur dari data yang
  dipotong di tanggal t.
- **Label market-neutral.** Target = return residual (dikurangi beta × S&P 500, dibagi volatilitas) yang diranking
  per tanggal — model memilih saham yang *mengungguli*, bukan yang sekadar ber-beta tinggi.
- **Uji signifikansi.** Setiap backtest melaporkan alpha/beta, benchmark beta-matched, CI Sharpe bootstrap, dan
  Deflated Sharpe Ratio yang memotong bukti sesuai jumlah konfigurasi yang dicoba.
- **Earnings & biaya.** Tidak membuka posisi bila rilis laba diperkirakan jatuh dalam masa pegang, atau bila
  ekspektasi return tidak menutup biaya pulang-pergi.
- **Walk-forward + gerbang promosi** (AUC ≥ 0,52, std AUC ≤ 0,05, ECE ≤ 0,06, proxy trading positif ≥ 50% fold, tidak
  lebih buruk dari model aktif). Gagal → REJECTED; model pertama yang gagal menjadi BASELINE (`MODEL STATUS: WARN`).
- **Freshness.** Data STALE → semua BUY menjadi WAIT dan headline `NO TRADE`.

## Biaya & kapasitas

- Target **$0**: GitHub Actions (repo publik), Aiven Free (1 GB disk), Streamlit Community Cloud, dan sumber data
  gratis (Yahoo, Wikipedia, SEC, FINRA). Bukan janji gratis selamanya.
- Storage: ±600–680 MB untuk ±900 saham dengan retensi default (batas efektif 800 MB, retensi otomatis di 85%).
  Rincian dan opsi penghematan di [`docs/DATABASE_DECISION.md`](docs/DATABASE_DECISION.md).
- Histori minimum walk-forward ±4,7 tahun → default 6 tahun.

## Struktur proyek

```
.github/workflows/  daily · setup · database_health_check · fundamental_refresh · model_retrain · weekly_backtest ·
                    db_maintenance · tests · challenger_{train,predict,tests}
app/
  data/          providers (Yahoo, CSV, ProviderChain), us_universe (S&P dari Wikipedia), universe manager, ingestion,
                 validator, cleaner, tickers (BRK.B → BRK-B)
  database/      db.py (SQLite/PostgreSQL, UPSERT), providers.py (Aiven/…), schema.py (migrasi aditif), repository.py
  features/      indikator & fitur teknikal (bebas look-ahead)
  market/        TradingCalendar (America/New_York), market regime, sector strength
  models/        label, model factory, kalibrasi, ensemble, trainer, retrain + kebijakan promosi
  strategy/      setup detector, entry, stop loss, take profit, scoring, filter NO-TRADE, explain
  risk/          position sizing (USD, saham utuh, batas risiko & likuiditas), konsentrasi
  scanner/       ranking & daily scanner, analisis per saham, cakupan per saham
  backtest/      engine (eksekusi konservatif), metrik, walk-forward
  valuation/     sec_edgar (10-K/10-Q PIT), rasio, nilai wajar (rentang), margin of safety, value trap
  flows/         FINRA short volume (slot foreign flow versi BEI), fitur 5/20/60H + baseline 120H, skor
  accumulation/  OBV/ADL/CMF/MFI/RVOL/CLV/VWAP, skor akumulasi, distribusi
  research/      skor terpisah + 9 ranking, STOCK RESEARCH REPORT, evaluasi, data dashboard
  challengers/   LSTM · GRU · ARIMA-GARCH · TFT (terpisah dari model aktif)
  utils/         market_rules (tick $0,01, tanpa lot)
dashboard/app.py    Streamlit (baca-saja)
config/             app · sources · model · trading · research · holidays (NYSE) · universe.csv (snapshot S&P 500)
docs/               SETUP_GUIDE · ARCHITECTURE · DATA_SOURCES · CLOUD_OPERATIONS · DATABASE_DECISION · KNOWN_LIMITATIONS ·
                    RESEARCH_MODULES · CHALLENGER_MODELS · examples/
scripts/            create_dashboard_role.py (role read-only untuk dashboard)
tests/              251 test (SQLite + PostgreSQL)
```

## Keterbatasan (ringkas)

Lengkap di [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md). Terpenting: Yahoo tidak resmi dan punya
survivorship bias; FINRA short volume **bukan** foreign flow; parameter strategi diwarisi dari versi BEI dan belum
dikalibrasi ulang untuk pasar AS — bandingkan ML vs teknikal-saja vs S&P 500 di Weekly Backtest sebelum memercayai
sinyal; tanggal earnings tidak disaring otomatis.
