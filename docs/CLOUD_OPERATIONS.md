# Operasi Cloud — GitHub Actions + Aiven PostgreSQL (versi US)

Semua proses produksi berjalan di **GitHub-hosted runners** dan hasilnya disimpan di **Aiven for PostgreSQL (Free)**.
Laptop hanya untuk menulis kode, menjalankan test, dan debugging. Langkah pembuatan akun dan setup awal ada di
[SETUP_GUIDE.md](SETUP_GUIDE.md); dokumen ini untuk operasional sehari-hari.

---

## 1. Database Aiven

- Satu service PostgreSQL Free per organisasi Aiven: 1 CPU, 1 GB RAM, **1 GB disk**, `max_connections` 20, tanpa
  connection pooling, tanpa batas waktu. Region ditentukan Aiven.
- Koneksi: Aiven Console → service → **Overview** → *Quick connect* / *Connection information* → Service URI:
  ```
  postgres://avnadmin:<password>@pg-<nama>-<project>.<x>.aivencloud.com:<port>/defaultdb?sslmode=require
  ```
- Role **read-only** untuk dashboard (jalankan sekali **setelah** Initial Setup, sebagai `avnadmin`):
  ```sql
  CREATE ROLE dashboard_ro LOGIN PASSWORD '<password-kuat>';
  GRANT CONNECT ON DATABASE defaultdb TO dashboard_ro;
  GRANT USAGE ON SCHEMA public TO dashboard_ro;
  GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_ro;
  ALTER DEFAULT PRIVILEGES FOR ROLE avnadmin IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro;
  ```
- **Power-off karena tidak aktif.** Aiven bisa mematikan service Free yang tidak punya aktivitas berkelanjutan dan
  mengirim email lebih dulu. Daily + Database Health Check (Senin–Jumat) menjaga aktivitas. Bila tetap mati:
  console → service → **Power on**, tunggu status *Running*, lalu jalankan Daily secara manual.

## 2. GitHub Secrets dan Variables

**Settings → Secrets and variables → Actions.**

| Nama | Jenis | Wajib | Isi |
|---|---|---|---|
| `DATABASE_URL` | Secret | **Ya** | Service URI Aiven (user `avnadmin`) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `NOTIFY_WEBHOOK_URL` | Secret | Tidak | notifikasi ringkasan harian |
| `FOREIGN_FLOW_API_URL`, `FOREIGN_FLOW_API_KEY` | Secret | Tidak | API flow vendor (pengganti FINRA) |
| `SEC_USER_AGENT` | **Variable** | Sangat disarankan | `Nama Anda email@domain.com` — wajib menurut SEC; tanpa ini fundamental jatuh ke snapshot Yahoo |
| `PORTFOLIO_VALUE` | Variable | Tidak | modal dalam **USD** untuk position sizing (default `10000`) |
| `DB_SIZE_LIMIT_MB` | Variable | Tidak | kosongkan (otomatis 800 MB untuk Aiven Free) |
| `MODEL_STORAGE`, `REPORT_STORAGE` | Variable | Tidak | default `auto` (tabel database) |

Secret tidak pernah dicetak: GitHub menyamarkannya di log, `describe()` hanya menampilkan host, dan pesan error
koneksi hanya memuat nama kelas exception.

## 3. Workflow

Semua workflow punya tombol **Run workflow**, `timeout-minutes`, dan `concurrency`. Workflow yang menulis data
berbagi grup `market-pipeline`, sehingga tidak pernah menulis database bersamaan.

| Workflow | File | Jadwal (WIB) | Cron (UTC) | Tugas |
|---|---|---|---|---|
| Daily Market Pipeline | `daily.yml` | Sel–Sab 06:15 | `15 23 * * 1-5` | test → health → universe → ingest incremental → validasi → fitur → scan → cakupan → riset (SEC, FINRA, valuasi, akumulasi) → simpan → report |
| Database Health Check | `database_health_check.yml` | Sen–Jum 19:47 | `47 12 * * 1-5` | koneksi, provider, ukuran, **uji tulis**; juga menjaga Aiven tetap aktif |
| Fundamental Refresh | `fundamental_refresh.yml` | Sab 16:13 | `13 9 * * 6` | tuntaskan fundamental SEC yang jatuh tempo + hitung ulang valuasi |
| Model Retrain | `model_retrain.yml` | Sab 21:00 | `0 14 * * 6` | retrain hanya bila model > 30 hari; promosi hanya bila lolos gerbang |
| Weekly Backtest | `weekly_backtest.yml` | Min 23:00 | `0 16 * * 0` | walk-forward + track record live + evaluasi modul riset |
| Database Maintenance | `db_maintenance.yml` | tgl 1, 19:00 | `0 12 1 * *` | retensi + VACUUM (opsi dry-run / FULL) |
| Initial Setup | `setup.yml` | manual | — | migrasi, universe, histori 6 tahun, model awal, backtest |
| Challenger Train / Predict | `challenger_*.yml` | Sab 18:37 / setelah Daily | `37 11 * * 6` | model LSTM/GRU/ARIMA-GARCH/TFT (opsional) |
| Tests | `tests.yml` | setiap push / PR | — | ruff + pytest di SQLite **dan** PostgreSQL 16 |

### Zona waktu dan kalender

- Cron GitHub ditulis dalam **UTC**. Pasar AS memakai daylight saving: tutup 16:00 ET = 20:00 UTC (EDT, Mar–Nov) atau
  21:00 UTC (EST). Daily dijadwalkan 23:15 UTC sehingga selalu setelah data final (±18:00 ET) di kedua musim.
- Di dalam job `TZ=America/New_York`. Tanggal bursa yang diharapkan dihitung `TradingCalendar` dengan zona
  `America/New_York`, `data_ready_time` 18:00, libur dari `config/holidays.yaml`, dan hari tanpa data `^GSPC`.
- Untuk Anda di Indonesia: hasil Daily biasanya tersedia **pagi hari WIB** (±06:15 bila tepat waktu; GitHub sering
  terlambat beberapa jam). Bursa AS buka lagi 20:30/21:30 WIB, jadi masih ada waktu untuk menyiapkan order.
- Keterlambatan aman: ingestion mengejar tanggal tertinggal; data terlalu tua → `STALE` → semua BUY menjadi WAIT.

### Idempotensi, retry, dan pemulihan

| Mekanisme | Di mana |
|---|---|
| UPSERT dengan kunci unik (run ulang tidak membuat duplikat) | `Database.upsert`, semua tabel |
| Retry provider + fallback; satu ticker gagal tidak menghentikan run | `ProviderChain`, `run_ingestion` |
| > 50% ticker gagal → `FAILED`; sebagian → `PARTIAL_SUCCESS` | `jobs._ingest` |
| Universe parsial (Wikipedia gagal) → tidak ada deaktivasi | `UniverseManager.update` |
| Retry koneksi database (service baru dinyalakan) | `PostgresDatabase.connect` (6× untuk Aiven) |
| Pemulihan storage penuh sebelum menulis (retensi + VACUUM FULL) | `jobs.ensure_writable` |
| Exit code: 0 sukses, 1 gagal, 2 config, 3 health kritis | `main.py` |

## 4. Kuota dan biaya

**Target operasi $0 dengan paket gratis selama batas mencukupi — bukan janji gratis selamanya.**

| Layanan | Batas relevan (dicek 16 Okt 2026) | Pemakaian proyek (estimasi) |
|---|---|---|
| GitHub Actions, repo **publik** | gratis untuk runner standar | ±20–30 jam runner/bulan |
| GitHub Actions, repo **privat** (GitHub Free) | 2.000 menit/bulan | ±1.300–1.800 menit/bulan — **mepet**; disarankan repo publik |
| Aiven PostgreSQL Free | 1 GB disk, 1 CPU, 20 koneksi | ±600–680 MB (DATABASE_DECISION.md) |
| Yahoo Finance | tidak resmi, batas tidak dipublikasikan | ±900 saham/hari (batch 50) |
| SEC EDGAR | 10 request/detik, wajib User-Agent | ≤ 150 emiten/hari + Sabtu sisa antrean |
| FINRA CDN | file publik | 1 file/hari (backfill awal 300 file) |
| Streamlit Community Cloud | 1 app privat + app publik tanpa batas | 1 app |

Risiko layanan gratis:

- **GitHub menonaktifkan workflow terjadwal di repo publik setelah 60 hari tanpa aktivitas repo.** Aktifkan lagi di
  Actions → workflow → *Enable workflow*, atau lakukan satu commit.
- Aiven bisa mematikan service idle atau mengubah paket Free.
- Yahoo/Wikipedia/SEC/FINRA bisa mengubah format atau memblokir sementara; status tiap sumber terlihat di tabel
  `data_sources` dan dashboard → Data Health.

## 5. Backup dan ekspor

Backup Aiven termasuk paket Free, tetapi retensinya tidak disebutkan. Ekspor berkala data yang **tidak bisa
direkonstruksi** dari laptop (butuh `pg_dump` versi ≥ server):

```bash
pg_dump "$DATABASE_URL" --data-only -t predictions -t prediction_evaluations -t trading_signals \
        -t backtest_runs -t pipeline_runs -t financial_statements -Fc -f swing_us_backup_$(date +%Y%m%d).dump
```

Harga, indeks, FINRA, dan fitur bisa direkonstruksi dengan workflow Initial Setup.

## 6. Monitoring

| Yang dipantau | Di mana |
|---|---|
| Status setiap run | tab Actions; tabel `pipeline_runs` |
| Cakupan per saham | tabel `scan_coverage`, report `scan_coverage_YYYYMMDD.csv`, dashboard → Full-Market Coverage |
| Kesehatan database | Database Health Check (harian), `python main.py health --strict` |
| Ukuran database | health `DATABASE SIZE`, Aiven Console → Metrics → disk usage |
| Sumber data (Yahoo, SEC, FINRA) | tabel `data_sources`, dashboard → Data Health |
| Log penting | tabel `system_logs` (retensi 90 hari) |

## 7. Dashboard

Dashboard adalah web service terpisah yang hanya membaca database.

- **Streamlit Community Cloud** (gratis): main file `dashboard/app.py`, requirements `requirements-dashboard.txt`,
  Secrets `DATABASE_URL = "<Service URI dengan user dashboard_ro>"`. Workspace gratis hanya punya **1 app privat**;
  bila slot itu sudah dipakai dashboard versi BEI, jadikan app US publik (tidak ada kredensial yang terlihat, hanya
  hasil analisis) atau pakai Render.
- **Render** web service gratis (`render.yaml`, `Dockerfile.dashboard`, env `DATABASE_URL`). Instance gratis tidur bila
  tidak dibuka dan butuh beberapa puluh detik untuk bangun.
- Dashboard berhenti dengan pesan jelas bila tersambung ke Supabase/Neon.

## 8. Pemeriksaan lokal (opsional, dari laptop)

```bash
export DATABASE_URL='postgres://avnadmin:...@pg-...aivencloud.com:12345/defaultdb?sslmode=require'
python main.py health --no-provider --strict   # koneksi, provider, ukuran, uji tulis
python main.py db-maintenance --dry-run        # apa yang akan dihapus retensi
```
