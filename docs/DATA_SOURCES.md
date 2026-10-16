# Sumber Data (versi US)

Semua sumber default **gratis dan otomatis** — tidak ada unduhan manual harian. Kredensial (bila ada) hanya lewat
GitHub Secrets/Variables.

## Ringkasan

| Data | Sumber utama | Cadangan | Frekuensi | Point-in-time? |
|---|---|---|---|---|
| Universe (daftar saham + sektor GICS) | Wikipedia: konstituen S&P 500 & S&P MidCap 400 | Dataset GitHub `datasets/s-and-p-500-companies` (S&P 500 saja) → `config/universe.csv` | tiap Daily | Ya (status per tanggal di `scan_coverage`) |
| Harga harian OHLCV | Yahoo Finance (`yfinance`) | CSV kanonik | tiap Daily | Ya |
| Indeks acuan | S&P 500 (`^GSPC`, Yahoo) | — | tiap Daily | Ya |
| Corporate action (split/dividen) | Yahoo | — | tiap Daily + scan penuh mingguan | Ya |
| Laporan keuangan | **SEC EDGAR** companyfacts (10-K/10-Q) | CSV kanonik → snapshot Yahoo | incremental 150 saham/hari + Fundamental Refresh Sabtu | **Ya** (tanggal `filed`) |
| Flow harian | **FINRA** Reg SHO daily short sale volume (CNMS) | CSV/API vendor | tiap Daily | Ya (terbit ≤ 18:00 ET hari D) |
| Hari libur bursa | `config/holidays.yaml` (NYSE 2018–2027) + dipelajari dari data `^GSPC` | — | perbarui tiap tahun | — |

## Detail dan keterbatasan per sumber

### Universe (`app/data/us_universe.py`)

- Default `universe.indexes: [sp500, sp400]` ≈ 900 saham — ukuran yang muat di database 1 GB.
- Wikipedia diperbarui komunitas setelah S&P mengumumkan perubahan indeks (biasanya efektif beberapa hari kemudian).
- **Pengaman:** bila salah satu indeks gagal diambil, hasil ditandai parsial dan **tidak ada saham yang dinonaktifkan**
  pada run itu. Deaktivasi juga ditolak bila daftar baru < 80% jumlah aktif.
- Saham yang keluar dari indeks → `is_active = false`; historinya tetap tersimpan.
- Kelas saham dinormalkan ke format Yahoo: `BRK.B` → `BRK-B`.

### Harga harian (Yahoo Finance)

- Sumber tidak resmi; kualitas/ketersediaan bisa berubah sewaktu-waktu.
- OHLC disesuaikan split secara retroaktif. Bila close pertama yang baru berbeda > 20% dari close terakhir di
  database (`data.split_refetch_jump`), histori ticker itu diunduh ulang.
- Saham yang delisting/diakuisisi biasanya hilang dari Yahoo → **survivorship bias** di backtest lama.

### Laporan keuangan (SEC EDGAR — `app/valuation/sec_edgar.py`)

- API publik `data.sec.gov`, tanpa API key. **Wajib** header User-Agent berisi nama + email Anda: GitHub Variable
  `SEC_USER_AGENT` (mis. `Budi Santoso budi@email.com`). Batas SEC 10 request/detik; sistem memakai ≤ 8.
- Angka sebuah periode diambil dari filing yang **pertama kali** melaporkannya; `publication_date` = tanggal `filed`.
  Restatement di filing berikutnya tidak bocor ke masa lalu (diuji di `tests/test_sec_edgar.py`).
- Arus kas 10-Q bersifat year-to-date → dipecah menjadi kuartal diskret; Q4 = FY − YTD 9 bulan.
- Hanya 10-K/10-Q. Emiten asing pelapor 20-F/40-F (jarang di S&P 500/400) jatuh ke cadangan Yahoo.
- Tag XBRL berbeda antar emiten/tahun; daftar tag cadangan dipakai. Bank sering tidak punya tag "Revenues" → revenue
  UNAVAILABLE (bukan nol); valuasi bank memakai justified P/B, laba, dan dividen.

### Flow harian (FINRA short volume — `app/flows/foreign_flow_provider.py`)

- `https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt`, tersedia sejak 1 Agustus 2018.
- Padanan AS untuk modul **foreign flow** versi BEI (AS tidak mempublikasikan beli/jual investor asing).
- Hanya mencakup volume yang **dilaporkan ke FINRA** (off-exchange/TRF), bukan volume konsolidasi; rasio short
  struktural ±50% karena aktivitas market maker. Karena itu skor memakai simpangan dari baseline 120 hari bursa
  saham itu sendiri. Ini **indikator tekanan short**, bukan bukti siapa yang membeli.
- Unduhan pertama: 300 hari bursa terakhir (±14 bulan, ±150 MB unduhan, beberapa menit). Selanjutnya hanya beberapa
  hari terakhir per run.

### Intraday

Tidak diimplementasikan (`BLOCKED_BY_DATA`) — data intraday gratis untuk ±900 saham tidak muat di database 1 GB dan
syarat penggunaan sumber gratis umumnya membatasi pengambilan massal.

## Fitur yang membutuhkan data eksternal

| Fitur | Tanpa data tambahan | Dengan data tambahan |
|---|---|---|
| Valuasi PIT untuk backtest | Ya (SEC EDGAR) | — |
| Flow beli/jual sungguhan | Tidak (FINRA = tekanan short) | CSV/API vendor + `foreign_flow.baseline_days: null` |
| Broker summary | Tidak ada padanan publik di AS | — |
| Data saham delisting (bebas survivorship) | Tidak | Feed berlisensi via CSV kanonik |
