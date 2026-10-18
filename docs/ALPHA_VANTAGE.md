# Alpha Vantage — memakai 25 request/hari sebaik mungkin

Paket gratis Alpha Vantage memberi **25 request per hari** (5 per menit). Untuk universe ±900 saham itu sangat
sedikit, jadi setiap request diarahkan ke data yang (1) tidak tersedia gratis dari SEC, Yahoo, atau FINRA, dan
(2) punya bukti empiris sebagai sinyal atau pengendali risiko.

## Apa yang diambil, dan kenapa

| Fungsi | Request | Untuk apa | Dasar |
|---|---|---|---|
| `EARNINGS_CALENDAR` | 1/hari | Tanggal rilis laba **resmi** berikutnya untuk seluruh pasar AS dalam 1 CSV, plus konsensus EPS kuartal depan. Filter `earnings_within_holding` memakai tanggal pasti, bukan lagi perkiraan dari pola SEC. | 52% exit di audit adalah stop-out; gap earnings melompati stop. |
| `EARNINGS` | ±20/hari (sisa kuota) | EPS aktual vs estimasi analis per kuartal, seluruh riwayat dalam 1 request. Menjadi fitur `f_sue`, `f_eps_surprise_pct`, dan `f_beat_streak`. | PEAD: Bernard & Thomas (1989), Livnat & Mendenhall (2006); kombinasi dengan reaksi harga: Chan, Jegadeesh & Lakonishok (1996). |
| `TREASURY_YIELD` 10Y/2Y/3M | ±1/hari | Risk-free valuasi otomatis (10Y menggantikan angka manual) dan kurva yield 10Y−3M di dashboard. | Kurva 10Y−3M: prediktor resesi klasik (Estrella & Mishkin 1998). |
| `FEDERAL_FUNDS_RATE` | ±1/bulan | Konteks kebijakan moneter (dashboard). | — |
| `NEWS_SENTIMENT` | 2/hari | Skor sentimen per ticker dari ±2.000 artikel terbaru. Mengisi komponen sentimen skor akhir yang sebelumnya selalu kosong di produksi. | Tetlock (2007); Heston & Sinha (2017). |

### Urutan antrean surprise EPS
1. **Segar**: saham yang baru rilis laba (8-K SEC atau kalender ≤ 10 hari) tetapi surprise kuartal itu belum ada.
   Drift paling kuat di minggu-minggu awal setelah rilis.
2. **Baru**: saham yang belum pernah diambil (S&P 500 dulu, lalu S&P 400). Backfill penuh ±45 hari bursa.
3. **Basi**: sisanya, yang paling lama tidak dicek (> 120 hari).

Kuota dihitung dari tabel `api_usage` dalam **jendela 24 jam bergulir** di semua workflow, jadi aman apa pun jam
reset Alpha Vantage. Respons "rate limit" (dikirim dengan HTTP 200) langsung menghentikan sisa panggilan run itu.

## Kejujuran data (point-in-time)

- **Surprise EPS** baru dianggap diketahui pada hari rilis untuk *pre-market*, dan hari bursa berikutnya untuk
  *post-market* atau jam tidak diketahui. Berlaku 91 hari kalender.
- **Kalender rilis** disimpan sebagai *log perubahan*: baris baru hanya bila tanggal atau estimasi berubah. Riwayat
  revisi konsensus mulai terkumpul sejak hari pertama. Hanya filter yang memakainya; fitur model
  `f_days_to_earn_est` tetap perkiraan SEC supaya training dan scan live konsisten.
- **Berita**: sebelum pengumpulan dimulai, jumlah berita bernilai *tidak diketahui* (NaN), bukan 0.
- **Treasury**: valuasi historis memakai yield yang sudah terbit pada tanggal itu (bukan nilai hari ini).

## Kapan fitur baru masuk model

Pemilihan fitur hanya memakai fitur dengan **cakupan > 50%** baris training. Akibatnya:
- `f_sue` dan kawan-kawan otomatis masuk setelah backfill menutup separuh universe (±25 hari bursa), pada retrain
  bulanan berikutnya, atau saat Anda menjalankan Model Retrain dengan `force=true`.
- Fitur berita (`f_news_*`) baru bisa masuk setelah riwayatnya mencakup separuh periode training (bertahun-tahun).
  Sampai saat itu berita hanya memengaruhi skor sentimen (bobot 5%) dan tampilan di dashboard.

Cek kontribusinya di **Weekly Backtest → Factor IC**: `f_sue` dan `f_xs_sue` dengan |t| ≥ 2 berarti sinyalnya
nyata di data Anda.

## Setup

1. Buat key gratis di https://www.alphavantage.co/support/#api-key.
2. GitHub repo → Settings → Secrets and variables → Actions → **New repository secret**:
   `ALPHAVANTAGE_API_KEY`. Jangan menulis key di file mana pun; repo ini publik.
3. Selesai. Langkah "Alpha Vantage" di workflow Daily, Weekly Backtest, dan Fundamental Refresh mulai berjalan.
   Tanpa secret, semua langkah itu dilewati dengan rapi.

Pantau di Data Health → **ALPHA VANTAGE** (kuota 24 jam, cakupan surprise EPS, tanggal kalender, yield, dan berita
terakhir), atau di annotation run Actions (judul "Alpha Vantage").

Perintah manual: `python main.py av-update [--mode daily|backfill] [--max-calls N]`.

## Yang sengaja TIDAK dipakai

- **Harga harian**: Yahoo sudah gratis tanpa batas. Harga adjusted dan histori penuh di Alpha Vantage berbayar.
- **Indikator teknikal**: dihitung sendiri dari harga.
- **Laporan keuangan**: SEC EDGAR sudah point-in-time dan gratis tanpa kuota.
- **Opsi (implied volatility/skew)** dan **insider per saham**: butuh 1 request per saham per hari, jauh di atas
  kuota. Insider bisa diambil gratis dari SEC Form 4 bila kelak dibutuhkan.
- **Intraday/realtime**: premium.
