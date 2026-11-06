# Protokol tuning (ditetapkan sebelum run tuning, 2026-09-21)

- Split: hanya aset tuning (`car`, `suzanne`, `spot`). Aset evaluasi tidak dipakai.
- Kendala: `--no-preserve-seams` untuk semua metode (setara dengan Decimate yang tidak mengunci seam).
- Grid: lima set bobot kanal × λ ∈ {0, 1, 2, 3, 5, 8, 12, 20}, rasio 0.5 / 0.2 / 0.05, sampel 4000, raster 128, seed 20260921.
  - W1 default: saliency 0.35, silhouette 0.40, sharp 0.25
  - W2 silhouette saja: silhouette 1
  - W3 saliency saja: saliency 1
  - W4 silhouette dominan: saliency 0.2, silhouette 0.6, sharp 0.2
  - W5 tanpa sharp: saliency 0.5, silhouette 0.5
- Kriteria seleksi: maksimalkan rata-rata ΔIoU 13-view (PERCEPTUAL − QEM) pada rasio 0.2 dan 0.05,
  dengan syarat rata-rata perubahan relatif Chamfer global ≤ +5 % pada kedua rasio.
  Pemecah seri: ROI Chamfer lebih kecil. Jika tidak ada konfigurasi dengan ΔIoU > 0 yang memenuhi syarat,
  laporkan apa adanya dan pilih λ terkecil yang tidak melanggar syarat.

## Amandemen 1 (21 September 2026, sebelum evaluasi apa pun dijalankan)

Run tuning pertama (arsip: `archive_pre_review/`, terpilih W4 dengan λ = 2) dibatalkan karena review kode adversarial
menemukan masalah yang mengubah metode dan metrik:

- kanal silhouette terkena aliasing sampling 24 view dan bergantung pada orientasi aset (diganti bobot kontur halus, 64 view);
- engine dapat menghasilkan face yang terbalik terhadap segitiga asalnya (ditambah penjaga `source_flip`, berlaku sama untuk QEM
  dan PERCEPTUAL);
- frame IoU silhouette sebelumnya berbeda untuk tiap kandidat (kini berbasis referensi) dan ROI kini diukur dua arah;
- saliency memuncak palsu pada tepi mesh terbuka (diperbaiki).

Grid bobot, nilai λ, rasio, sampel, seed, split, dan kriteria seleksi **tidak diubah**. Satu-satunya perubahan protokol:
resolusi raster silhouette dinaikkan dari 128 ke **512 px** untuk menekan noise aliasing IoU (±2·10⁻⁴ pada 128 px, setara
dengan besar efek yang diamati). Hasil run ulang disimpan di folder `W*/` dan `selection.txt` yang baru.
