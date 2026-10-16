# Keterbatasan yang Diketahui (versi US)

Sistem ini adalah **alat riset dan pendukung keputusan**, bukan sistem profit terjamin. Tidak ada klaim akurasi
atau profit yang sudah dibuktikan. Semua hasil uji lokal memakai data **sintetis** (`make-sample`), sehingga tidak
mengatakan apa pun tentang pasar saham AS.

## 1. Data harga dan universe

- **Yahoo Finance tidak resmi**: tanpa SLA dan lisensi, bisa berubah atau diblokir kapan saja.
- **Survivorship bias.** Yahoo tidak menyediakan saham yang sudah delisting/diakuisisi, dan universe berbasis
  konstituen indeks HARI INI. Saham yang dulu ada di S&P 500/400 lalu keluar tidak ikut di backtest masa lalu,
  jadi backtest historis cenderung terlalu optimistis dalam derajat yang tidak diketahui.
- Universe dari Wikipedia (dirawat komunitas). Kesalahan edit sesaat bisa terjadi; pengaman parsial/rasio 80%
  mencegah deaktivasi massal, tetapi satu-dua saham bisa keliru sampai halaman diperbaiki.
- Hari libur NYSE di `config/holidays.yaml` mencakup 2018–2027; **tambahkan 2028 dst.** dari nyse.com. Penutupan
  mendadak (mis. hari berkabung nasional) dipelajari otomatis dari hari tanpa data `^GSPC`.
- Hari setengah hari (tutup 13:00 ET) diperlakukan sebagai hari bursa biasa.

## 2. Eksekusi dan biaya

- Backtest memasukkan fee beli 0,10%, fee jual 0,11% (asumsi broker/aplikasi untuk investor Indonesia, termasuk
  spread kurs; SEC Section 31 $20,60 per $1 juta mulai 4 April 2026), slippage 5 bps, tick $0,01, saham utuh,
  limit order t+1, gap pada stop, dan stop-first bila SL dan TP tersentuh di candle yang sama.
- **Belum dimodelkan:** spread bid-ask aktual, jeda LULD, perdagangan pra/pasca-pasar (gap earnings terjadi di
  luar jam reguler), pajak dividen 15–30% untuk investor non-AS (W-8BEN), dan selisih kurs USD/IDR atas modal.
- **Earnings:** saham AS melaporkan kuartalan dengan gap besar. Sistem tidak menyaring tanggal earnings secara
  eksplisit; cek kalender earnings sebelum masuk posisi swing.

## 3. Scalping dan intraday — `BLOCKED_BY_DATA`

- Tidak ada sumber intraday, order book, atau trade-by-trade yang layak dan gratis untuk ±900 saham.
- GitHub Actions tidak cocok untuk eksekusi intraday: run terjadwal bisa terlambat berjam-jam dan interval cron
  minimum 5 menit.
- Database 1 GB tidak cukup untuk bar intraday seluruh universe.

## 4. Fundamental dan valuasi

- SEC EDGAR memberi data point-in-time, tetapi hanya 10-K/10-Q. Emiten asing (20-F/40-F) memakai snapshot Yahoo.
- Pemetaan tag XBRL memakai daftar cadangan; emiten dengan tag khusus bisa punya field UNAVAILABLE (bukan nol).
- `max_years: 6` → histori valuasi maksimal ±6 tahun.
- Asumsi valuasi (risk-free 5,1%, ERP 4,25%, g 3% di `config/research.yaml`) harus ditinjau manual. Health check
  memberi WARN bila lebih dari 180 hari.

## 5. Flow dan broker

- **FINRA short volume bukan foreign flow.** Ini volume jual-short yang dilaporkan ke fasilitas FINRA (off-exchange),
  termasuk aktivitas market maker yang netral arah. Dipakai sebagai simpangan relatif terhadap kebiasaan saham itu
  sendiri; jangan dibaca sebagai "institusi sedang membeli".
- Short interest (posisi short terbuka, dua kali sebulan) belum dipakai.
- Broker summary tidak punya padanan publik di AS.
- Indikator akumulasi berbasis harga-volume (OBV, ADL, CMF, MFI) adalah **perkiraan**, bukan bukti transaksi aktual.

## 6. Model dan strategi

- Parameter strategi (setup, ATR stop, ambang regime) diwarisi dari versi BEI dan belum dikalibrasi ulang untuk
  pasar AS. Bandingkan strategi ML vs teknikal-saja vs **benchmark beta-matched** di Weekly Backtest, dan lihat
  verdict signifikansi (QUANT_V2.md §2.5) sebelum memercayai sinyal.
- Audit 16 Okt 2026: backtest produksi (6 bulan, Sharpe 0,47) secara statistik **belum** membuktikan edge — standard
  error Sharpe untuk 6 bulan ±1,4. Butuh ≥ 2–3 tahun out-of-sample untuk bukti yang layak.
- `f_ear3` adalah proksi kejutan laba dari reaksi harga, bukan surprise EPS vs konsensus analis (data konsensus tidak
  tersedia gratis). Tanggal rilis berikutnya **diperkirakan** dari pola kuartalan sebelumnya — bisa meleset beberapa
  hari bila emiten menggeser jadwal; tetap cek kalender earnings untuk posisi besar.
- Bukan portofolio long-short: label residual memilih saham yang relatif lebih kuat, tetapi posisi tetap long-only
  sehingga return masih ikut beta pasar.
- Pada data sintetis, lapisan ML tidak konsisten mengungguli aturan teknikal saja.
- Bila model pertama gagal gerbang promosi, sistem mengaktifkannya sebagai BASELINE dan health check menampilkan
  `MODEL STATUS: WARN`.
- Long-term engine belum punya horizon, label, dan backtest jangka panjang sendiri.

## 7. Operasional cloud

- Jadwal GitHub Actions tidak presisi (bisa terlambat beberapa jam). Tanggal bursa tetap dihitung dengan jam New York.
- **Workflow terjadwal di repo publik dinonaktifkan GitHub setelah 60 hari tanpa aktivitas repo.**
- **Aiven Free bisa di-power-off** bila tidak ada aktivitas berkelanjutan; nyalakan ulang di console. Backup
  termasuk paket tetapi retensinya tidak disebutkan → ekspor berkala (CLOUD_OPERATIONS §5).
- Repo privat memakan kuota menit GitHub Actions (2.000 menit/bulan pada akun Free).

## 8. Verifikasi saat porting (16 Okt 2026)

- Test suite lulus di SQLite **dan** PostgreSQL 16 lokal (264 lulus setelah v2; 10 dilewati karena butuh torch/statsmodels untuk
  model challenger — dijalankan di workflow `challenger_tests.yml`).
- Pipeline `make-sample → setup → daily → research` berjalan end-to-end di SQLite dan PostgreSQL.
- Lingkungan pengembangan tidak bisa mengakses Yahoo, Wikipedia, SEC, maupun FINRA, jadi pengambilan data **nyata**
  baru terverifikasi setelah workflow Initial Setup dan Daily hijau di GitHub Actions Anda. Parser Wikipedia, SEC,
  dan FINRA diuji dengan fixture berformat sama dengan sumber aslinya.
