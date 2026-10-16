# Keputusan Database Produksi (versi US)

**Keputusan: Aiven for PostgreSQL — paket Free, di akun/organisasi Aiven milik Anda, khusus proyek ini.**

Batas paket gratis di bawah ini dicek dari halaman resmi pada **16 Oktober 2026**. Vendor bisa mengubahnya kapan
saja, jadi periksa ulang sebelum mengandalkannya.

## Mengapa bukan Supabase atau Neon

Pemilik repo sudah memakai **Supabase** (aplikasi lain) dan **Neon** (versi BEI proyek ini). Versi US tidak boleh
menumpang di sana. Kode menegakkan aturan ini:

- `APP_ENV=production` + `DATABASE_URL` ber-host Supabase **atau** Neon → job berhenti **sebelum** koneksi dan migrasi
  (`app/pipeline/context.guard_production_database`, exit 2). Override sadar hanya lewat
  `ALLOW_SUPABASE_DATABASE=true` / `ALLOW_NEON_DATABASE=true`.
- Dashboard yang tersambung ke Supabase/Neon menampilkan error dan berhenti (`database_banner`).

## Perbandingan (paket gratis tanpa kartu kredit)

| Kriteria | **Aiven Free** | Koyeb Free | Render Free | Prisma Postgres Free | Neon / Supabase |
|---|---|---|---|---|---|
| Storage | 1 GB disk | 1 GB | 1 GB | — | — |
| Compute | 1 CPU, 1 GB RAM, **tanpa batas jam** | 0,25 vCPU, **5 jam compute/bulan** | 0,1 CPU | kuota operasi bulanan | — |
| Idle | bisa di-power-off bila tidak ada aktivitas berkelanjutan (ada email notifikasi); nyalakan ulang di console | tidur setelah 5 menit | — | — | — |
| Umur | tanpa batas waktu | tanpa batas | **dihapus setelah 30 hari** (+14 hari tenggang) | — | — |
| Koneksi | `max_connections` 20, tanpa pooling | ±112 | — | — | — |
| Status | **dipilih** | ditolak: 5 jam/bulan habis oleh job harian (±15–25 menit × 22 hari + ekor idle) | ditolak: bukan penyimpanan jangka panjang | ditolak: kuota operasi tidak cocok untuk UPSERT ribuan baris per hari | **dilarang** (dipakai proyek lain) |

Sumber: [Aiven PG free tier](https://aiven.io/docs/products/postgresql/concepts/pg-free-tier),
[Aiven free plan](https://aiven.io/docs/platform/concepts/free-plan), [Koyeb databases](https://www.koyeb.com/docs/databases),
[Render free](https://render.com/docs/free).

### Alasan memilih Aiven

1. **Tidak ada batas jam compute.** Job batch harian (Daily ±15–25 menit, Fundamental Refresh, Retrain, Backtest)
   tidak menghabiskan kuota waktu seperti Koyeb.
2. **Permanen.** Tidak kedaluwarsa seperti Render.
3. **PostgreSQL standar dengan TLS** — kode hanya butuh `DATABASE_URL`; profil Aiven sudah ada di
   `app/database/providers.py` (deteksi `*.aivencloud.com`, retry koneksi 6×, `sslmode=require`).

### Risiko Aiven yang diterima (dan mitigasinya)

- **Power-off karena tidak aktif.** Aiven tidak menyebut durasinya. Mitigasi: Daily (Senin–Jumat) dan
  **Database Health Check** (Senin–Jumat, uji tulis) menjaga aktivitas. Bila tetap mati, Anda menerima email; buka
  console → service → **Power on**. Job yang gagal konek tampil merah di tab Actions (GitHub mengirim email).
- **1 GB disk termasuk WAL dan file sistem.** Batas efektif yang dipakai sistem 800 MB (dibulatkan ke bawah). Retensi
  otomatis berjalan di 85%, health check WARN di 80% dan FAIL di 95%.
- **Satu layanan PostgreSQL gratis per organisasi.** Jangan dipakai bersama proyek lain.
- **Region tidak bisa dipilih** pada paket Free (ditentukan Aiven).
- **Backup** termasuk paket, tetapi retensinya tidak disebutkan → ekspor CSV berkala hal yang penting (lihat
  CLOUD_OPERATIONS §5). Harga dan indeks selalu bisa diunduh ulang lewat `setup`; yang tidak bisa direkonstruksi
  adalah riwayat prediksi, evaluasi, dan registry model.

## Kesesuaian storage

Ukuran per baris diukur di PostgreSQL 16 (versi BEI, ±920 emiten); universe US default ±900 saham (S&P 500 + 400).

| Tabel | Pertumbuhan | Dengan retensi default versi US |
|---|---|---|
| `price_history` | ±45 MB/tahun | 6 tahun ≈ 270 MB |
| `predictions` + evaluasi | ±180 MB/tahun | WAIT/AVOID 120 hari ≈ 60 MB + BUY/WATCHLIST (disimpan selamanya, ±10%) |
| `foreign_flow_history` (FINRA short volume) | ±45 MB/tahun | 2 tahun ≈ 90 MB |
| `financial_statements` (SEC) | ±7 MB/tahun | 6 tahun periode ≈ 40 MB (tidak pernah dihapus) |
| `valuation_results` | ±22 MB/tahun | ±30 MB + akhir bulan |
| `scan_coverage` | ±28 MB/tahun | 120 hari ≈ 10 MB |
| `model_forecasts` (challenger, opsional) | ±50 MB/tahun | 90 hari ≈ 25 MB |
| model & report | — | model aktif + 3 versi; report 60 hari |

Perkiraan total ±600–680 MB, di bawah batas efektif 800 MB. **Bila mendekati batas**, pilihan berurutan:
1. turunkan `retention.price_history_years` ke 5 (minimum walk-forward);
2. turunkan `research_retention.foreign_flow_years` ke 1;
3. matikan workflow challenger (`model_forecasts`);
4. persempit universe ke `indexes: [sp500]` (±500 saham).

Menambah S&P 600 (`indexes: [sp500, sp400, sp600]`, ±1.500 saham) **tidak muat** di 1 GB.

## Implementasi

| Provider | Deteksi host | Batas efektif (`DB_SIZE_LIMIT_MB=auto`) | SSL | Retry koneksi | Produksi |
|---|---|---|---|---|---|
| aiven | `*.aivencloud.com` | 800 MB (dari 1 GB disk) | `sslmode=require` | 6× | **dipakai** |
| neon | `*.neon.tech` | 450 MB | wajib | 6× | ditolak (override `ALLOW_NEON_DATABASE`) |
| supabase | `*supabase.co/.com` | 450 MB | wajib | 3× | ditolak (override `ALLOW_SUPABASE_DATABASE`) |
| postgres lain | host lain | 1000 MB (set `DB_SIZE_LIMIT_MB`) | `require` | 4× | boleh |

Setelah upgrade paket, set Variable repo `DB_SIZE_LIMIT_MB` ke batas baru (MB).

Skema didefinisikan sekali di `app/database/schema.py` dan dimigrasi secara **additive** (`CREATE … IF NOT EXISTS`,
`ALTER TABLE ADD COLUMN`), dicatat di `schema_migrations`. Migrasi tidak pernah menghapus tabel atau kolom.
Nama tabel/kolom sama dengan versi BEI (mis. `foreign_flow_history` kini berisi FINRA short volume) agar kode,
skor, dan dashboard tetap identik.
