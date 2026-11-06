# Pra-registrasi analisis evaluasi (dikunci sebelum run evaluasi, 2026-09-21)

Dokumen ini ditulis setelah tuning selesai dan **sebelum** metode apa pun dijalankan pada aset evaluasi.
Tujuannya mencegah pemilihan metrik, rasio, atau uji setelah melihat hasil.

## Konfigurasi tetap (dari `results/tuning`, aturan di `results/tuning/PROTOCOL.md`)

- Bobot kanal PERCEPTUAL: saliency 0.2, silhouette 0.6, sharp 0.2 (visibility tidak aktif; tanpa kamera).
- λ = 2, mode `quadric` (weighted quadrics).
- Hasil seleksi tuning (n = 3 aset, rasio 0.2 dan 0.05): ΔIoU 13-view +0.00113, Chamfer +2.8 %,
  Hausdorff −7.5 %, ROI Chamfer −0.5 %. Angka ini hanya bukti tuning, bukan hasil penelitian.

## Protokol run

- Aset: split `evaluation` di `dataset/manifest.json` (9 aset; 7 memiliki `LOD_EvalROI`).
- `--no-preserve-seams` untuk semua metode (kendala sama dengan Decimate). Kondisi dengan seam terkunci tidak dijalankan
  di evaluasi primer; bila dijalankan kemudian, dilaporkan sebagai analisis sekunder terpisah.
- Rasio 0.5 / 0.2 / 0.05; sampel 4000; raster 128; seed 20260921; timing 1 warm-up + 3 ulangan dengan urutan metode diacak.
- Kondisi: main (DECIMATE, QEM, PERCEPTUAL), ablation, sweep λ ∈ {0, 1, 2, 3, 5, 8, 12, 20} dengan bobot di atas.
- Tanpa mask proteksi manual (kondisi otomatis). `LOD_EvalROI` hanya untuk evaluasi.

## Hipotesis primer (keluarga Holm berisi 2 uji)

Pada rasio **0.05** (LOD3), PERCEPTUAL dibanding **QEM** (baseline terkendali yang berbagi engine):

- **H1 (silhouette):** IoU silhouette 13-view PERCEPTUAL lebih tinggi. Uji: Wilcoxon signed-rank dua sisi, eksak, n = 9 aset.
- **H2 (area penting):** ROI Chamfer (referensi → LOD, ternormalisasi) PERCEPTUAL lebih rendah. Uji: Wilcoxon signed-rank dua sisi,
  eksak, n = 7 aset yang memiliki ROI.

Koreksi Holm–Bonferroni atas H1 dan H2; α = 0.05. Dengan n = 7, p eksak dua sisi terkecil adalah 2/128 = 0.0156, sehingga H2 hanya dapat
signifikan setelah Holm bila arah efek konsisten pada hampir semua aset. Ini keterbatasan yang diterima dan dilaporkan.

## Analisis sekunder / eksploratori (tanpa klaim konfirmatori)

- Semua rasio lain, perbandingan dengan DECIMATE, Chamfer dan Hausdorff global (biaya yang diperkirakan), non-ROI Chamfer,
  IoU 6-sumbu, waktu pemrosesan, ablation, dan kurva λ.
- Ukuran efek (rank-biserial), selisih berpasangan dengan CI bootstrap per aset, serta win/tie/loss per aset dan per kategori.
- Hasil yang tidak mendukung H1/H2 tetap dilaporkan lengkap.

## Amandemen 1 (21 September 2026, sebelum run evaluasi dimulai)

Konfigurasi di atas **dibatalkan** karena review kode adversarial (37 temuan terverifikasi) mengubah metode dan metrik:
kanal silhouette (bobot kontur halus, 64 view), penjaga `source_flip` pada engine, frame IoU berbasis referensi,
ROI dua arah, perbaikan saliency di tepi terbuka, serta interval kepercayaan yang lebih jujur untuk n kecil.
Tuning diulang dengan grid dan kriteria yang sama (`results/tuning/PROTOCOL.md`, Amandemen 1); arsip run lama ada di
`results/tuning/archive_pre_review/`.

**Konfigurasi tetap yang baru** (seleksi mekanis, `results/tuning/selection.txt`):

- Bobot kanal PERCEPTUAL: silhouette 1 (saliency 0, sharp 0, visibility tidak aktif), λ = 12, mode `quadric`,
  silhouette 64 view pada raster 256 px.
- Bukti tuning (n = 3, rasio 0.2 dan 0.05, vs QEM): ΔIoU 13-view +0.00158 (error silhouette −11.2 %), Chamfer +4.6 %,
  Hausdorff tersampel +38.2 %, ROI Chamfer (ref→LOD) +10.3 %. Saliency saja memperburuk IoU dan ROI pada semua λ.

**Perubahan protokol evaluasi:**

- Raster metrik silhouette 512 px (sebelumnya 128 px; noise aliasing ±2·10⁻⁴ setara besar efek).
- H2 memakai **ROI Chamfer simetris** (`roi_chamfer_sym_n`; rerata dua arah) karena ukuran satu arah tidak melihat
  artefak pada hasil (temuan review #19). H1 tetap IoU 13-view.
- Keluarga Holm primer dinyatakan eksplisit: `--holm-metrics iou_multiview,roi_chamfer_sym_n --holm-baselines QEM
  --holm-ratios 0.05` (2 uji). Interval kepercayaan selisih berpasangan: Hodges–Lehmann dengan CI eksak Walsh
  (bukan bootstrap persentil); CI tidak dilaporkan untuk n < 5.
- Sweep λ memakai bobot silhouette-saja di atas. Run eksploratori terpisah dengan bobot default lama
  (saliency 0.35, silhouette 0.40, sharp 0.25) disimpan di folder berbeda dan tidak dipakai untuk memilih apa pun.
- Pengukuran waktu dijalankan terpisah dan berurutan (tanpa proses paralel lain) agar tidak terganggu beban CPU;
  run kualitas boleh paralel per kelompok aset karena hasilnya deterministik.

**Prediksi yang dicatat sebelum melihat data evaluasi:** berdasarkan bukti tuning, H1 mungkin didukung, sedangkan
**H2 kemungkinan tidak didukung** (bobot silhouette memindahkan anggaran segitiga dari interior/wajah ke outline).
Hasil apa pun dilaporkan apa adanya.

**Eksperimen sekunder (deskriptif, tidak konfirmatori): bantuan mask manual.** Pada tiga aset `character_face`
(max_planck, igea, ogre), PERCEPTUAL dijalankan dengan mask manual = `LOD_EvalROI` untuk mensimulasikan pengguna
mengecat area wajah dengan fitur *Paint Important Areas*, pada λ = 12 dan rasio yang sama. Karena area yang dilindungi
sama dengan area yang diukur, eksperimen ini sirkular secara desain. Tujuannya hanya menunjukkan bahwa kontrol manual
bekerja dan berapa biayanya di luar area tersebut, bukan membuktikan kemampuan persepsi otomatis.

## Catatan kontaminasi yang diketahui

- `stanford-bunny` (aset evaluasi) dipakai selama pengembangan v0.2 untuk profil kecepatan engine/rasterizer dan pemeriksaan visual
  heatmap saliency. Aset ini tidak dipakai untuk memilih λ atau bobot. Parameter `noise_floor` saliency (0.02) dipilih oleh pengembang
  dari nilai kontras beberapa mesh uji; sumber lengkapnya dicatat di kode `smart_lod/perception.py`.
