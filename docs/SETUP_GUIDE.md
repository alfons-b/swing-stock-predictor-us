# Panduan Setup Lengkap — swing-stock-predictor-us

Panduan ini membawa Anda dari folder proyek sampai sistem berjalan otomatis setiap hari bursa AS, lengkap dengan
dashboard. Semua layanan yang dipakai **gratis** dan **tidak memakai Supabase maupun Neon**.

| Komponen | Layanan | Biaya |
|---|---|---|
| Penjadwal & komputasi | GitHub Actions (repo publik) | gratis |
| Database | **Aiven for PostgreSQL — Free** (1 GB disk) | gratis, tanpa kartu kredit |
| Dashboard | Streamlit Community Cloud (alternatif: Render) | gratis |
| Data harga | Yahoo Finance | gratis |
| Daftar saham + sektor | Wikipedia (S&P 500 + S&P MidCap 400) | gratis |
| Laporan keuangan | SEC EDGAR | gratis |
| Flow harian | FINRA daily short sale volume | gratis |

Perkiraan waktu: ±45–90 menit (sebagian besar menunggu workflow selesai).

---

## 0. Yang perlu disiapkan

- Akun **GitHub** (Anda sudah punya: `alfons-b`).
- Email aktif (untuk Aiven, dan untuk header wajib SEC).
- **Git** di laptop (`git --version` untuk mengecek). Python 3.12 di laptop hanya perlu untuk Langkah 8 (role dashboard)
  dan development lokal — jalannya sistem sendiri tidak butuh laptop.
- Zip proyek `swing-stock-predictor-us.zip`.

> **Penting:** jangan pakai ulang `DATABASE_URL` dari versi BEI (Neon) atau Supabase. Versi US **menolak berjalan**
> di produksi bila host database-nya Neon/Supabase — ini disengaja agar tidak menyentuh database proyek lain.

---

## Langkah 1 — Buat repository GitHub dan push kode

1. Buka **github.com/new**.
   - *Repository name:* `swing-stock-predictor-us`
   - *Visibility:* **Public** (disarankan). Repo publik mendapat menit GitHub Actions gratis tanpa batas; repo privat
     dibatasi 2.000 menit/bulan dan sistem ini memakai ±1.300–1.800 menit/bulan (mepet). Kode tidak berisi secret.
   - **Jangan** centang *Add a README*, *.gitignore*, atau *license* (repo harus kosong).
   - Klik **Create repository**.
2. Ekstrak zip, lalu push dari laptop.

   **Windows (PowerShell)** — jalankan satu per satu:
   ```powershell
   cd C:\dev
   Expand-Archive .\swing-stock-predictor-us.zip -DestinationPath .
   cd swing-stock-predictor-us
   git init
   git add .
   git commit -m "swing-stock-predictor-us: versi pasar AS"
   git branch -M main
   git remote add origin https://github.com/alfons-b/swing-stock-predictor-us.git
   git push -u origin main
   ```
   **macOS / Linux:**
   ```bash
   unzip swing-stock-predictor-us.zip && cd swing-stock-predictor-us
   git init && git add . && git commit -m "swing-stock-predictor-us: versi pasar AS"
   git branch -M main
   git remote add origin https://github.com/alfons-b/swing-stock-predictor-us.git
   git push -u origin main
   ```
   Simpan proyek di luar folder OneDrive (mis. `C:\dev\`) agar tidak bentrok dengan sinkronisasi file.
3. Buka tab **Actions** di repo. Bila muncul tombol *I understand my workflows, go ahead and enable them*, klik.
   Push pertama otomatis menjalankan workflow **Tests** — tunggu sampai hijau (±5–8 menit).

---

## Langkah 2 — Buat database Aiven (PostgreSQL Free)

1. Buka **console.aiven.io** → *Sign up*. Bisa langsung dengan akun Google/GitHub. **Tidak perlu kartu kredit.**
2. Saat diminta, buat *organization* dan *project* (mis. `swing-us`).
3. Di project: **Services → Create service → PostgreSQL®**.
4. Pilih **Service tier: Free**. Pada paket Free, cloud dan region ditentukan Aiven (tidak bisa dipilih).
5. *Service name:* `swing-us-pg` → **Create service**.
6. Tunggu status berubah dari *Rebuilding* menjadi **Running** (±2–5 menit).
7. Buka service → **Overview** → *Quick connect* (atau *Connection information*) → salin **Service URI**. Bentuknya:
   ```
   postgres://avnadmin:<password>@pg-xxxxxxx-swing-us.x.aivencloud.com:12345/defaultdb?sslmode=require
   ```
   Simpan di password manager. URI ini berisi password admin.

Catatan Aiven Free (1 CPU, 1 GB RAM, **1 GB disk**, maks. 20 koneksi, tanpa batas waktu):
- Satu service PostgreSQL gratis per organisasi — jangan dipakai bersama proyek lain.
- Service yang **tidak aktif** bisa di-*power-off* oleh Aiven (ada email pemberitahuan). Workflow harian sistem ini
  menjaganya tetap aktif. Bila suatu saat mati: console → service → **Power on**.

---

## Langkah 3 — Isi GitHub Secrets dan Variables

Di repo: **Settings → Secrets and variables → Actions**.

**Tab *Secrets* → New repository secret:**

| Name | Value | Wajib |
|---|---|---|
| `DATABASE_URL` | Service URI Aiven dari Langkah 2.7 | **Ya** |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | lihat Langkah 9 | Tidak |

**Tab *Variables* → New repository variable:**

| Name | Value | Wajib |
|---|---|---|
| `SEC_USER_AGENT` | `Alfons Bahy alfonsbahy73@gmail.com` (nama + email Anda) | **Sangat disarankan** — SEC mewajibkan header ini. Tanpa ini, laporan keuangan jatuh ke snapshot Yahoo yang bukan point-in-time. |
| `PORTFOLIO_VALUE` | modal Anda dalam **USD**, mis. `5000` | Tidak (default `10000`) |
| `DB_SIZE_LIMIT_MB` | kosongkan | Tidak (otomatis 800 MB untuk Aiven Free) |

`SEC_USER_AGENT` adalah *Variable* (bukan secret) karena isinya memang dikirim ke SEC sebagai identitas Anda.

---

## Langkah 4 — Pastikan workflow Tests hijau

**Actions → Tests**. Bila run dari push pertama sudah hijau, lanjut. Bila belum ada, klik **Run workflow**.
Tests menjalankan lint, lalu pytest di SQLite dan di PostgreSQL 16 sungguhan (service container).

---

## Langkah 5 — Initial Setup (sekali saja)

**Actions → Initial Setup → Run workflow** (biarkan *Lewati backtest awal* tidak dicentang) → **Run workflow**.

Yang dilakukan (±20–40 menit):
1. migrasi skema database (24 tabel);
2. mengambil universe: konstituen **S&P 500 + S&P 400** dari Wikipedia (±900 saham) beserta sektor GICS;
3. mengunduh **6 tahun** harga harian + S&P 500 (`^GSPC`) dari Yahoo, plus split/dividen;
4. validasi data, membangun fitur;
5. walk-forward (4 fold) + perbandingan model, menyimpan `model_v001` sebagai **ACTIVE**;
6. backtest awal (validation + test).

Hasil sukses di halaman *Summary* run:
```
Setup selesai. Model model_v001: PROMOTED            (atau PROMOTED_AS_BASELINE)
Data: +1.3xx.xxx baris, 9xx emiten, gagal <beberapa>
Backtest validation: return ... vs S&P 500 ...
Backtest test: return ... vs S&P 500 ...
```
- `PROMOTED_AS_BASELINE` artinya model pertama belum lolos gerbang kualitas; tetap dipakai agar Daily bisa jalan, dan
  health check menampilkan `MODEL STATUS: WARN`. Filter NO-TRADE tetap berlaku.
- Beberapa ticker gagal itu normal (mis. ticker yang baru berganti kode). Run gagal hanya bila > 50% ticker gagal.

Bila run gagal: buka log step yang merah. Penyebab paling umum ada di bagian **Troubleshooting**. Initial Setup
**idempoten** — aman dijalankan ulang; hanya data yang belum ada yang diunduh.

---

## Langkah 6 — Database Health Check

**Actions → Database Health Check → Run workflow**. Harus hijau dengan baris-baris berikut:
```
DATABASE:            OK  (postgres[aiven]:pg-...aivencloud.com:12345/defaultdb)
DATABASE PROVIDER:   OK  (aiven ...)
DATABASE SIZE:       OK  (xxx MB dari 800 MB ...)
DATABASE WRITE:      OK
UNIVERSE:            OK  (9xx saham aktif (sumber: konstituen indeks sp500, sp400 via Wikipedia; cadangan universe.csv))
```

---

## Langkah 7 — Daily pertama (manual)

**Actions → Daily Market Pipeline → Run workflow**.

Run pertama lebih lama (±25–45 menit) karena sekali saja mengunduh **300 hari file FINRA short volume** dan laporan
SEC untuk 150 saham pertama. Run berikutnya ±15–25 menit.

Hasil ada di *Summary* run (report Markdown) dan di artifact `daily-report-<id>` (HTML/CSV). Contoh baris utama:
```
2026-10-16 | Regime BULL | Data UP_TO_DATE | Model model_v001
TOP 3 SWING OPPORTUNITIES
 # Ticker Setup               Skor P(bull)            Entry        SL       TP1       TP2   R/R   Qty Decision
 1 XYZ    BREAKOUT_VOLUME       71     41%   123.40-125.10    117.85    133.20    139.90  1.95     8 BUY
```
`Qty` = jumlah saham (tanpa lot) sesuai `PORTFOLIO_VALUE` dan risiko 1% per trade. Bila hasilnya
**NO TRADE / NO HIGH-CONVICTION SETUP**, itu jawaban yang sah — sistem tidak memaksa sinyal.

Laporan keuangan SEC terisi bertahap (150 saham/hari + sisa antrean tiap Sabtu lewat Fundamental Refresh), jadi
valuasi seluruh universe lengkap dalam ±1 minggu. Untuk mempercepat: **Actions → Fundamental Refresh → Run workflow**.

---

## Langkah 8 — Dashboard

### 8a. Buat role read-only (sekali, dari laptop)

Dashboard sebaiknya tidak memakai user admin. Aiven tidak punya SQL editor bawaan, jadi pakai skrip di repo:

**Windows (PowerShell):**
```powershell
cd C:\dev\swing-stock-predictor-us
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # bila diblokir: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
$env:DATABASE_URL = "postgres://avnadmin:...@pg-....aivencloud.com:12345/defaultdb?sslmode=require"
python scripts\create_dashboard_role.py
```
**macOS / Linux:**
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
DATABASE_URL='postgres://avnadmin:...' python scripts/create_dashboard_role.py
```
Masukkan password baru (min. 12 karakter). Skrip mencetak baris `DATABASE_URL = "postgres://dashboard_ro:...` —
salin untuk langkah berikut. Jalankan skrip **setelah** Initial Setup (tabel harus sudah ada).

### 8b. Deploy di Streamlit Community Cloud

1. Buka **share.streamlit.io** → login dengan GitHub → **Create app** → *Deploy a public app from GitHub*.
2. *Repository:* `alfons-b/swing-stock-predictor-us` · *Branch:* `main` · *Main file path:* `dashboard/app.py`.
3. **Advanced settings:**
   - *Python version:* **3.12**
   - *Secrets:*
     ```toml
     DATABASE_URL = "postgres://dashboard_ro:<password>@pg-....aivencloud.com:12345/defaultdb?sslmode=require"
     ```
4. **Deploy**. Dependensi dibaca otomatis dari `dashboard/requirements.txt`.

Workspace Streamlit gratis hanya punya **1 app privat** (app publik tidak terbatas). Bila slot privat sudah dipakai
dashboard versi BEI, biarkan app US publik (yang terlihat hanya hasil analisis, bukan kredensial) atau pakai Render:
**render.com → New → Blueprint** → pilih repo ini (`render.yaml`) → isi env `DATABASE_URL`. Instance Render gratis tidur
bila tidak dibuka dan butuh beberapa puluh detik untuk bangun.

Halaman dashboard: Market Overview, Full-Market Coverage, Top Swing Stocks, Sector Strength, Stock Detail,
Undervalued Screener, **Short Flow (FINRA)**, Accumulation / Distribution, Integrated Research, Prediction History,
Backtest, Model Performance, Model Lab (challenger), Pipeline Status, Data Health.

---

## Langkah 9 (opsional) — Notifikasi Telegram

1. Di Telegram, chat **@BotFather** → `/newbot` → ikuti instruksi → salin *token*.
2. Kirim pesan apa saja ke bot baru Anda.
3. Buka di browser `https://api.telegram.org/bot<TOKEN>/getUpdates` → cari `"chat":{"id": ...}` → itu *chat id*.
4. Tambahkan Secrets `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID`. Ringkasan Daily akan terkirim setelah tiap run.

Alternatif: Secret `NOTIFY_WEBHOOK_URL` (webhook Slack/Discord).

---

## Jadwal otomatis

Semua jadwal ditulis dalam UTC; konversi ke WIB tetap sepanjang tahun (WIB tidak memakai DST). Bursa AS tutup 16:00 ET
= 03:00 WIB (Mar–Nov) / 04:00 WIB (Nov–Mar).

| Workflow | Kapan (WIB) | Fungsi |
|---|---|---|
| Daily Market Pipeline | Selasa–Sabtu ±06:15 | scan hari bursa AS sebelumnya |
| Database Health Check | Senin–Jumat 19:47 | cek database + uji tulis (menjaga Aiven aktif) |
| Fundamental Refresh | Sabtu 16:13 | tuntaskan laporan SEC + valuasi ulang |
| Challenger Train | Sabtu 18:37 | model LSTM/GRU/TFT/ARIMA-GARCH (opsional) |
| Model Retrain | Sabtu 21:00 | retrain bila model > 30 hari |
| Weekly Backtest | Minggu 23:00 | walk-forward + evaluasi modul |
| Database Maintenance | tanggal 1, 19:00 | retensi + VACUUM |

Jadwal GitHub **tidak presisi** — run sering terlambat 1–6 jam. Bursa AS baru buka lagi 20:30/21:30 WIB, jadi
hasil Daily tetap tersedia jauh sebelum sesi berikutnya. Butuh lebih cepat? Jalankan Daily manual kapan saja.

---

## Rutinitas Anda

| Kapan | Apa |
|---|---|
| Tiap hari bursa (pagi WIB) | Buka dashboard → Top Swing Stocks. **Cek kalender earnings** saham kandidat sebelum masuk posisi (tidak disaring otomatis). |
| Mingguan | Dashboard → Backtest: bandingkan strategi ML vs teknikal-saja vs S&P 500 buy & hold. |
| Bila dapat email "workflow failed" | Buka run yang merah; lihat Troubleshooting. |
| Bila dapat email Aiven "service powered off" | console.aiven.io → service → **Power on**, lalu jalankan Daily manual. |
| Tiap ±6 bulan | Perbarui asumsi valuasi di `config/research.yaml` (UST 10Y, ERP; health WARN bila > 180 hari). |
| Tiap akhir tahun | Tambahkan libur NYSE tahun berikutnya ke `config/holidays.yaml` (sumber: nyse.com/markets/hours-calendars). |
| Bila repo tidak di-commit 60 hari | GitHub menonaktifkan jadwal di repo publik — aktifkan lagi di tab Actions atau lakukan satu commit. |

---

## Menyesuaikan

Semua lewat file config (commit + push) atau Variable GitHub:

| Ingin mengubah | Di mana |
|---|---|
| Modal (USD) | Variable `PORTFOLIO_VALUE` |
| Risiko per trade (default 1%) | `config/trading.yaml` → `portfolio.RISK_PER_TRADE` |
| Fee broker untuk backtest | `config/trading.yaml` → `backtest.TRANSACTION_FEE_BUY/SELL` (default 0,10%/0,11%; broker AS bebas komisi tanpa konversi kurs: `0.0` / `0.00003`) |
| Universe | `config/sources.yaml` → `universe.indexes` (`[sp500]` lebih hemat; `sp600` tidak muat di 1 GB) |
| Harga minimum / likuiditas | `config/app.yaml` → `quality.min_price` ($5), `liquidity.MIN_AVG_TRADING_VALUE` ($10 juta/hari) |
| Jam Daily | `.github/workflows/daily.yml` → `cron` (UTC) |

---

## Development lokal (opsional)

Uji tanpa internet dengan data **sintetis**:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
$env:YAHOO_ENABLED = "false"; $env:FINRA_ENABLED = "false"
python main.py make-sample
python main.py setup
python main.py --as-of 2026-10-02 daily
python -m pytest -q
```
**macOS / Linux:**
```bash
export YAHOO_ENABLED=false FINRA_ENABLED=false
python main.py make-sample && python main.py setup && python main.py --as-of 2026-10-02 daily
pytest -q
```
Tanpa `DATABASE_URL`, development memakai `sqlite:///data/local.db`. Variabel di `.env` **tidak** dibaca otomatis;
set lewat shell. Opsi global `--as-of` ditulis **sebelum** nama perintah.

Uji dengan data nyata di laptop (butuh internet): hapus dua variabel di atas, set `SEC_USER_AGENT`, lalu
`python main.py setup` dan `python main.py daily` (SQLite lokal).

---

## Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `DATABASE_URL menunjuk ke Neon/Supabase ...` (exit 2) | secret masih URL database proyek lain | ganti secret `DATABASE_URL` dengan Service URI Aiven |
| `connection refused` / `timeout` ke `*.aivencloud.com` | service Aiven mati (power-off) atau masih *Rebuilding* | console Aiven → **Power on**, tunggu *Running*, jalankan ulang |
| `too many connections` | dashboard + job + laptop bersamaan melebihi 20 koneksi | tutup koneksi laptop; jalankan ulang |
| `UNIVERSE: WARN ... DAFTAR CONTOH` | Wikipedia & GitHub gagal diakses dan memakai `universe.example.csv` | jalankan ulang Daily; cek log "Konstituen ... gagal" |
| Log "Universe PARSIAL ... TIDAK dinonaktifkan" | salah satu halaman Wikipedia gagal | tidak perlu tindakan; run berikutnya memperbaiki |
| `FUNDAMENTALS: WARN` + `sec_edgar NOT_CONFIGURED` | Variable `SEC_USER_AGENT` kosong / tanpa `@` | isi Variable `SEC_USER_AGENT` |
| `SHORT FLOW: WARN FOREIGN_FLOW_UNAVAILABLE` | FINRA belum diunduh (sebelum Daily pertama) atau CDN gagal | jalankan Daily; cek `data_sources` di dashboard → Data Health |
| `DATA FRESHNESS: STALE` | Daily belum jalan sejak hari bursa terakhir / Yahoo gagal | jalankan Daily manual |
| `DATABASE SIZE: WARN/FAIL` | mendekati 800 MB | Actions → Database Maintenance (dry_run=false, full=true); lihat opsi hemat di DATABASE_DECISION.md |
| `Setup ... butuh ±4,7 tahun data` | `initial_history_years` terlalu kecil | biarkan default 6 |
| Workflow terjadwal tidak jalan | repo tidak aktif 60 hari / workflow di-*disable* | Actions → pilih workflow → *Enable workflow* |
| Dashboard error "Supabase/Neon" | Secret dashboard salah | ganti `DATABASE_URL` di Secrets Streamlit → Reboot app |
| Dashboard `permission denied` | role read-only dibuat sebelum tabel ada | jalankan ulang `scripts/create_dashboard_role.py` |

---

## Checklist akhir

- [ ] Repo `swing-stock-predictor-us` dibuat (publik) dan kode ter-push
- [ ] Workflow **Tests** hijau
- [ ] Service Aiven PostgreSQL Free *Running*
- [ ] Secret `DATABASE_URL` (Aiven) + Variable `SEC_USER_AGENT` terisi
- [ ] **Initial Setup** hijau (model_v001 aktif)
- [ ] **Database Health Check** hijau (`DATABASE PROVIDER: OK (aiven)`, `DATABASE WRITE: OK`)
- [ ] **Daily Market Pipeline** manual pertama hijau
- [ ] Role `dashboard_ro` dibuat dan dashboard ter-deploy
- [ ] (Opsional) Notifikasi Telegram

Setelah semua tercentang, sistem berjalan sendiri. Bacaan lanjutan: [CLOUD_OPERATIONS.md](CLOUD_OPERATIONS.md)
(operasional), [DATA_SOURCES.md](DATA_SOURCES.md), [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

> Sistem ini alat riset dan pendukung keputusan, **bukan nasihat investasi**. Probabilitas berasal dari model
> historis dan bisa salah.
