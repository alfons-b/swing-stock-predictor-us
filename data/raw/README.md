# data/raw — data input riset (opsional)

Versi US mengambil semua data riset utama **otomatis dan gratis** (SEC EDGAR untuk laporan keuangan, FINRA untuk
short volume), jadi folder ini **boleh kosong**. File yang Anda commit ke repo dibaca oleh workflow GitHub Actions
setelah `checkout`. Jangan menyimpan secret di sini.

| Folder | Isi | Format |
|---|---|---|
| `fundamentals/` | Laporan keuangan dari sumber lain (mis. vendor berlisensi) — **mengalahkan Yahoo, kalah dari SEC** | CSV/XLSX kanonik — template: `python main.py fundamentals-template` |
| `foreign_flow_csv/` | Data flow alternatif dari vendor (beli/jual sungguhan) | CSV: `date,ticker,foreign_buy_shares,foreign_sell_shares[,total_volume_shares,...]` — set `foreign_flow.baseline_days: null` |
| `fx/` | Kurs untuk laporan non-USD (jarang di S&P 500/400) | `fx_rates.csv`: `date,currency,rate` (1 unit mata uang = `rate` USD) |
| `broker_summary/` | Tidak ada padanan publik di AS (nonaktif) | lihat `app/accumulation/broker_flow.py` |

Catatan:
- `publication_date` di file fundamental harus tanggal publikasi ASLI laporan. Bila tidak diketahui, kosongkan —
  sistem memakai tanggal pertama data diketahui (aman untuk live, tidak dipakai mundur di backtest).
