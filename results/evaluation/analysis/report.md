# Analisis hasil Smart LOD

Dibuat oleh `scripts/analyze_results.py` dari: `results/evaluation/quality_A`, `results/evaluation/quality_B`, `results/evaluation/quality_C`, `results/evaluation/timing`. Semua angka dihitung dari `raw.jsonl`; `environment.json` hanya untuk informasi mesin.

## 1. Data

- Record: 486 quality, 243 timing (+81 warm-up, tidak dianalisis), 18 asset, 0 error (0 tanpa hasil pengganti). Baris rusak/tidak dikenal: 0; masalah lain (berkas hilang, pengaturan/kode berbeda): 0; duplikat yang diganti record terakhir: 0.
- Surface samples per mesh: [4000]; metric seed: [20260921]; silhouette resolution: [512].
- λ of main PERCEPTUAL: [12]; weighting: ['quadric']; ratios: ['50%', '20%', '5%'].
- Run `33777b8a8eef` (completed): Blender 5.1.2, NumPy 2.3.4, 12th Gen Intel(R) Core(TM) i7-12700H, 20 logical cores, Windows-11-10.0.26300-SP0, extension 0.2.0; 2026-09-21T06:23:22+00:00 → 2026-09-21T06:40:01+00:00; sessions: 1.
- Run `0d97a443e317` (completed): Blender 5.1.2, NumPy 2.3.4, 12th Gen Intel(R) Core(TM) i7-12700H, 20 logical cores, Windows-11-10.0.26300-SP0, extension 0.2.0; 2026-09-21T06:23:22+00:00 → 2026-09-21T06:42:09+00:00; sessions: 1.
- Run `510b3cb2c293` (completed): Blender 5.1.2, NumPy 2.3.4, 12th Gen Intel(R) Core(TM) i7-12700H, 20 logical cores, Windows-11-10.0.26300-SP0, extension 0.2.0; 2026-09-21T06:23:22+00:00 → 2026-09-21T06:43:54+00:00; sessions: 1.
- Run `dc43a0bae884` (completed): Blender 5.1.2, NumPy 2.3.4, 12th Gen Intel(R) Core(TM) i7-12700H, 20 logical cores, Windows-11-10.0.26300-SP0, extension 0.2.0; 2026-09-21T06:43:56+00:00 → 2026-09-21T07:02:17+00:00; sessions: 1.

| asset | category | split | triangles | ROI vertices | locked vertices | harness wall s |
|---|---|---|---:|---:|---:|---:|
| beetle | hard_surface | evaluation | 49,272 | 3,112 | 0 | 324.9 |
| fandisk | hard_surface | evaluation | 12,946 | — | 0 | 129.2 |
| rocker_arm | hard_surface | evaluation | 20,088 | — | 0 | 175.4 |
| max_planck | character_face | evaluation | 99,991 | 9,171 | 0 | 670 |
| igea | character_face | evaluation | 100,000 | 4,433 | 0 | 674.6 |
| ogre | character_face | evaluation | 100,000 | 21,836 | 0 | 642.1 |
| bunny | organic | evaluation | 69,451 | 8,132 | 0 | 513.3 |
| armadillo | organic | evaluation | 99,976 | 5,418 | 0 | 694.5 |
| horse | organic | evaluation | 96,966 | 4,896 | 0 | 635.9 |

### Pemeriksaan integritas

- Repeated QEM pass identical: 9/9 assets.
- λ = 0 control equals QEM: 54/54 rows.
- Snapshot hash equals manifest geometry_sha256: 9/9 assets.
- ROI vertex count equals manifest: 9/9 assets.
- Timing runs reproduce the quality-run arrays: 108/108 rows.

### Anggaran segitiga tidak sama (>2 %) atau target tidak tercapai (main)

- Tidak ada.

## 2. Batas interpretasi

- Uji berpasangan memakai paling banyak n = 9 aset (split `evaluation`; metrik ROI hanya aset ber-ROI). p eksak terkecil yang mungkin (tanpa ties) = 2/2^n. Famili Holm (metrik iou_multiview, roi_chamfer_sym_n; baseline QEM; rasio 5%) berisi 2 uji, ambang terkecil 0.05/2 = 0.025.
- p Holm terbaik yang mungkin per metrik famili (bila setiap uji famili mencapai p eksak terkecilnya): iou_multiview (n = 9, p min 0.00391) ≥ 0.00781; roi_chamfer_sym_n (n = 7, p min 0.0156) ≥ 0.0156. Signifikansi setelah Holm secara prinsip mungkin untuk setiap metrik famili, tetapi n tetap kecil. Laporkan ukuran efek, CI dan W/T/L, bukan hanya p.
- CI = interval eksak Walsh/Hodges–Lehmann untuk pergeseran berpasangan (inversi uji Wilcoxon; tingkat tercapai di kolom `CI level`, mis. 96,1 % pada n = 9 dan 95,3 % pada n = 7). Tidak dilaporkan untuk n < 5; pada n = 5 interval 95 % tidak mungkin. Kolom `boot_ci_*` di tabel CSV adalah bootstrap persentil rata-rata yang nominal dan terlalu sempit pada n sekecil ini (cakupan sekitar 0,89–0,91 pada n = 9 untuk data normal); jangan dikutip sebagai CI 95 %.
- Sapuan λ dan ablation pada split `evaluation` bersifat eksploratori (analisis sekunder). Jangan memilih atau mengubah λ/bobot dari hasil ini; penanda Pareto sengaja tidak dihitung di luar split `tuning`. Penalaan ulang setelah melihat data ini harus dilaporkan sebagai analisis eksploratori baru.
- Metode berbeda dapat berhenti pada jumlah segitiga aktual yang berbeda (kunci seam/UV, batas topologi). Bandingkan kurva kualitas–segitiga dan daftar anggaran di bawah; hasil yang lebih padat bukan bukti metode lebih baik.
- Ada trade-off: λ yang menaikkan IoU silhouette dapat menaikkan Hausdorff/Chamfer global. Laporkan kedua arah; front Pareto hanya deskriptif.
- Chamfer/Hausdorff adalah estimasi berbasis sampel acak (seed tetap); `hausdorff_n` hanya dari sampel dan dapat meremehkan artefak lokal, `hausdorff_v_n` menambahkan setiap vertex kedua mesh. IoU memakai 6 atau 13 view ortografik pada raster tetap. ROI `LOD_EvalROI` dianotasi secara geometris oleh satu orang dan hanya dipakai untuk evaluasi; `roi_*_n` hanya referensi → LOD (tidak melihat geometri berlebih di ROI), `roi_*_sym_n` dua arah.
- Waktu adalah wall-clock pada satu mesin; implementasi Python (QEM/PERCEPTUAL) vs C (Decimate) mencerminkan implementasi, bukan kompleksitas algoritmik. Pengulangan waktu bukan sampel independen antar-aset. Waktu QEM/PERCEPTUAL mencakup konversi mesh sumber ke larik dan kunci seam/material (kolom terpisah); waktu DECIMATE tidak mencakup pembacaan LOD ke larik (hanya untuk pengukuran; kolom terpisah).
- Asosiasi antar metrik tidak menunjukkan sebab-akibat; ablation hanya menunjukkan perubahan pada data ini.

## 3. PERCEPTUAL vs baseline, berpasangan per aset (split `evaluation`)

Δ = PERCEPTUAL − baseline dalam satuan metrik; `better` memberi arah yang baik (`lower`: Δ < 0 menguntungkan PERCEPTUAL, `higher`: Δ > 0). W/T/L dihitung dari sisi PERCEPTUAL dengan toleransi seri 0.01×|baseline| (jarak, waktu) atau 0.0005 (IoU). r_rb > 0 berarti PERCEPTUAL lebih baik. HL Δ = estimasi Hodges–Lehmann; CI = interval eksak Walsh untuk pergeseran itu (tingkat tercapai di `CI level`; kosong bila n < 5 atau bila 95 % tidak tercapai). Holm atas 2 uji (metrik iou_multiview, roi_chamfer_sym_n; baseline QEM; rasio 5%); baris lain eksploratori (p Holm kosong). Kolom lengkap (median, perubahan relatif, metode p, bootstrap nominal): `tables/paired_main.md`.

| ratio | metric | better | vs | n | mean Δ | HL Δ | CI low | CI high | CI level | W | T | L | p | p Holm | r_rb | unequal tris | family |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50% | chamfer_n | lower | QEM | 9 | +2.67e-06 | +9.82e-07 | -1.87e-08 | +9.16e-06 | 0.9609 | 3 | 0 | 6 | 0.0547 | — | -0.733 | 0 | exploratory |
| 50% | chamfer_n | lower | DECIMATE | 9 | +2.7e-06 | +1.02e-06 | -5.57e-08 | +9.19e-06 | 0.9609 | 3 | 0 | 6 | 0.0547 | — | -0.733 | 0 | exploratory |
| 50% | hausdorff_n | lower | QEM | 9 | +0.000104 | +9.2e-05 | +2.21e-05 | +0.000197 | 0.9609 | 1 | 0 | 8 | 0.00781 | — | -0.956 | 0 | exploratory |
| 50% | hausdorff_n | lower | DECIMATE | 9 | +8.5e-05 | +7.01e-05 | -2.07e-06 | +0.000175 | 0.9609 | 1 | 0 | 8 | 0.0547 | — | -0.733 | 0 | exploratory |
| 50% | hausdorff_v_n | lower | QEM | 9 | +0.000106 | +0.000108 | -9.51e-06 | +0.000223 | 0.9609 | 3 | 1 | 5 | 0.109 | — | -0.667 | 0 | exploratory |
| 50% | hausdorff_v_n | lower | DECIMATE | 9 | +9.78e-05 | +0.000107 | -2.24e-05 | +0.000223 | 0.9609 | 2 | 2 | 5 | 0.0977 | — | -0.644 | 0 | exploratory |
| 50% | iou_axis | higher | QEM | 9 | +5.45e-05 | +5.35e-05 | +2.45e-05 | +8.43e-05 | 0.9609 | 0 | 9 | 0 | 0.00781 | — | +1 | 0 | exploratory |
| 50% | iou_axis | higher | DECIMATE | 9 | +5.35e-05 | +5.35e-05 | +2.42e-05 | +8.39e-05 | 0.9609 | 0 | 9 | 0 | 0.00781 | — | +1 | 0 | exploratory |
| 50% | iou_multiview | higher | QEM | 9 | +7.8e-05 | +7.2e-05 | +2.23e-05 | +0.000137 | 0.9609 | 0 | 9 | 0 | 0.00391 | — | +1 | 0 | exploratory |
| 50% | iou_multiview | higher | DECIMATE | 9 | +7.83e-05 | +7.22e-05 | +2.29e-05 | +0.000139 | 0.9609 | 0 | 9 | 0 | 0.00391 | — | +1 | 0 | exploratory |
| 50% | roi_chamfer_n | lower | QEM | 7 | +3.53e-06 | +3.62e-06 | -1.46e-06 | +8.67e-06 | 0.9531 | 2 | 0 | 5 | 0.156 | — | -0.643 | 0 | exploratory |
| 50% | roi_chamfer_n | lower | DECIMATE | 7 | +3.55e-06 | +3.65e-06 | -1.46e-06 | +8.67e-06 | 0.9531 | 2 | 0 | 5 | 0.156 | — | -0.643 | 0 | exploratory |
| 50% | roi_hausdorff_n | lower | QEM | 7 | +8.6e-05 | +7.11e-05 | +2.6e-05 | +0.000156 | 0.9531 | 0 | 0 | 7 | 0.0156 | — | -1 | 0 | exploratory |
| 50% | roi_hausdorff_n | lower | DECIMATE | 7 | +8.87e-05 | +7.56e-05 | +2.66e-05 | +0.000165 | 0.9531 | 0 | 0 | 7 | 0.0156 | — | -1 | 0 | exploratory |
| 50% | roi_chamfer_sym_n | lower | QEM | 7 | +3.52e-06 | +3.42e-06 | -7.99e-07 | +8.39e-06 | 0.9531 | 2 | 0 | 5 | 0.109 | — | -0.714 | 0 | exploratory |
| 50% | roi_chamfer_sym_n | lower | DECIMATE | 7 | +3.62e-06 | +3.6e-06 | -1.36e-06 | +8.48e-06 | 0.9531 | 2 | 0 | 5 | 0.156 | — | -0.643 | 0 | exploratory |
| 50% | roi_hausdorff_sym_n | lower | QEM | 7 | +0.000123 | +0.000116 | +4.01e-05 | +0.00024 | 0.9531 | 0 | 0 | 7 | 0.0156 | — | -1 | 0 | exploratory |
| 50% | roi_hausdorff_sym_n | lower | DECIMATE | 7 | +0.000126 | +0.000121 | +4.13e-05 | +0.00024 | 0.9531 | 0 | 0 | 7 | 0.0156 | — | -1 | 0 | exploratory |
| 50% | nonroi_chamfer_n | lower | QEM | 7 | +1.95e-07 | +3.88e-07 | -1.09e-06 | +1.26e-06 | 0.9531 | 3 | 0 | 4 | 0.812 | — | -0.143 | 0 | exploratory |
| 50% | nonroi_chamfer_n | lower | DECIMATE | 7 | +2.21e-07 | +3.94e-07 | -1.07e-06 | +1.31e-06 | 0.9531 | 3 | 0 | 4 | 0.812 | — | -0.143 | 0 | exploratory |
| 50% | time_s | lower | QEM | 9 | +4.19 | +4.23 | +3.21 | +5.28 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |
| 50% | time_s | lower | DECIMATE | 9 | +9.36 | +10.1 | +6.21 | +12.4 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |
| 20% | chamfer_n | lower | QEM | 9 | +5.65e-06 | +3.08e-06 | +5.07e-07 | +1.51e-05 | 0.9609 | 0 | 1 | 8 | 0.00781 | — | -0.956 | 0 | exploratory |
| 20% | chamfer_n | lower | DECIMATE | 9 | +5.88e-06 | +3.43e-06 | +4.58e-07 | +1.53e-05 | 0.9609 | 0 | 2 | 7 | 0.0117 | — | -0.911 | 0 | exploratory |
| 20% | hausdorff_n | lower | QEM | 9 | +0.000324 | +0.000328 | +0.00014 | +0.000513 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |
| 20% | hausdorff_n | lower | DECIMATE | 9 | +0.00031 | +0.00032 | +0.000127 | +0.000513 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |
| 20% | hausdorff_v_n | lower | QEM | 9 | +0.000272 | +0.000218 | -0.00013 | +0.000812 | 0.9609 | 2 | 0 | 7 | 0.129 | — | -0.6 | 0 | exploratory |
| 20% | hausdorff_v_n | lower | DECIMATE | 9 | +0.000267 | +0.000236 | -0.000136 | +0.000829 | 0.9609 | 2 | 0 | 7 | 0.203 | — | -0.511 | 0 | exploratory |
| 20% | iou_axis | higher | QEM | 9 | +0.000166 | +0.000138 | +5.01e-05 | +0.000337 | 0.9609 | 1 | 8 | 0 | 0.00391 | — | +1 | 0 | exploratory |
| 20% | iou_axis | higher | DECIMATE | 9 | +0.000142 | +0.000101 | +7.33e-06 | +0.000332 | 0.9609 | 1 | 8 | 0 | 0.0117 | — | +0.911 | 0 | exploratory |
| 20% | iou_multiview | higher | QEM | 9 | +0.000193 | +0.000168 | +6.23e-05 | +0.000347 | 0.9609 | 1 | 8 | 0 | 0.00391 | — | +1 | 0 | exploratory |
| 20% | iou_multiview | higher | DECIMATE | 9 | +0.000162 | +0.000151 | +3.12e-05 | +0.000303 | 0.9609 | 1 | 8 | 0 | 0.0273 | — | +0.822 | 0 | exploratory |
| 20% | roi_chamfer_n | lower | QEM | 7 | +9.13e-06 | +8.52e-06 | -4.24e-06 | +2.34e-05 | 0.9531 | 2 | 0 | 5 | 0.297 | — | -0.5 | 0 | exploratory |
| 20% | roi_chamfer_n | lower | DECIMATE | 7 | +9.2e-06 | +8.63e-06 | -3.6e-06 | +2.34e-05 | 0.9531 | 2 | 0 | 5 | 0.219 | — | -0.571 | 0 | exploratory |
| 20% | roi_hausdorff_n | lower | QEM | 7 | +0.000172 | +0.000175 | -5.73e-05 | +0.000418 | 0.9531 | 3 | 0 | 4 | 0.219 | — | -0.571 | 0 | exploratory |
| 20% | roi_hausdorff_n | lower | DECIMATE | 7 | +0.000185 | +0.000186 | -4.5e-05 | +0.000418 | 0.9531 | 2 | 0 | 5 | 0.156 | — | -0.643 | 0 | exploratory |
| 20% | roi_chamfer_sym_n | lower | QEM | 7 | +9.27e-06 | +8.86e-06 | -2.02e-06 | +2.21e-05 | 0.9531 | 3 | 0 | 4 | 0.219 | — | -0.571 | 0 | exploratory |
| 20% | roi_chamfer_sym_n | lower | DECIMATE | 7 | +8.54e-06 | +8.72e-06 | -2.04e-06 | +2.01e-05 | 0.9531 | 2 | 0 | 5 | 0.109 | — | -0.714 | 0 | exploratory |
| 20% | roi_hausdorff_sym_n | lower | QEM | 7 | +0.000174 | +0.000175 | -5.73e-05 | +0.000418 | 0.9531 | 3 | 0 | 4 | 0.219 | — | -0.571 | 0 | exploratory |
| 20% | roi_hausdorff_sym_n | lower | DECIMATE | 7 | +0.000178 | +0.000183 | -5e-05 | +0.000418 | 0.9531 | 2 | 0 | 5 | 0.156 | — | -0.643 | 0 | exploratory |
| 20% | nonroi_chamfer_n | lower | QEM | 7 | +1.12e-06 | +1.41e-06 | -2.11e-06 | +3.88e-06 | 0.9531 | 1 | 2 | 4 | 0.297 | — | -0.5 | 0 | exploratory |
| 20% | nonroi_chamfer_n | lower | DECIMATE | 7 | +1.21e-06 | +1.43e-06 | -1.92e-06 | +4.2e-06 | 0.9531 | 1 | 2 | 4 | 0.297 | — | -0.5 | 0 | exploratory |
| 20% | time_s | lower | QEM | 9 | +4.17 | +4.18 | +3.13 | +5.35 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |
| 20% | time_s | lower | DECIMATE | 9 | +11.9 | +12.9 | +7.64 | +15.9 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |
| 5% | chamfer_n | lower | QEM | 9 | +1.52e-05 | +7.49e-06 | -5.08e-06 | +5.44e-05 | 0.9609 | 3 | 0 | 6 | 0.203 | — | -0.511 | 0 | exploratory |
| 5% | chamfer_n | lower | DECIMATE | 9 | +1.74e-05 | +1.01e-05 | -3.64e-06 | +5.23e-05 | 0.9609 | 2 | 0 | 7 | 0.0977 | — | -0.644 | 0 | exploratory |
| 5% | hausdorff_n | lower | QEM | 9 | +0.000248 | +0.000474 | -0.00101 | +0.00102 | 0.9609 | 2 | 0 | 7 | 0.203 | — | -0.511 | 0 | exploratory |
| 5% | hausdorff_n | lower | DECIMATE | 9 | +0.00022 | +0.000503 | -0.00129 | +0.00107 | 0.9609 | 2 | 0 | 7 | 0.164 | — | -0.556 | 0 | exploratory |
| 5% | hausdorff_v_n | lower | QEM | 9 | -5.88e-05 | +9.64e-05 | -0.000847 | +0.00052 | 0.9609 | 4 | 1 | 4 | 0.734 | — | -0.156 | 0 | exploratory |
| 5% | hausdorff_v_n | lower | DECIMATE | 9 | +1.52e-05 | +0.000227 | -0.00102 | +0.00066 | 0.9609 | 3 | 0 | 6 | 0.359 | — | -0.378 | 0 | exploratory |
| 5% | iou_axis | higher | QEM | 9 | +0.000249 | +0.000177 | -0.000448 | +0.00095 | 0.9609 | 2 | 6 | 1 | 0.359 | — | +0.378 | 0 | exploratory |
| 5% | iou_axis | higher | DECIMATE | 9 | -0.000224 | -1.69e-05 | -0.00116 | +0.000478 | 0.9609 | 1 | 6 | 2 | 1 | — | -0.0222 | 0 | exploratory |
| 5% | iou_multiview | higher | QEM | 9 | +0.000529 | +0.000355 | +0.00018 | +0.000926 | 0.9609 | 3 | 6 | 0 | 0.00391 | 0.00781 | +1 | 0 | holm |
| 5% | iou_multiview | higher | DECIMATE | 9 | +0.000104 | +0.00026 | -0.000941 | +0.000764 | 0.9609 | 2 | 6 | 1 | 0.129 | — | +0.6 | 0 | exploratory |
| 5% | roi_chamfer_n | lower | QEM | 7 | +2.95e-05 | +2.54e-05 | -1.48e-05 | +8.6e-05 | 0.9531 | 2 | 1 | 4 | 0.297 | — | -0.5 | 0 | exploratory |
| 5% | roi_chamfer_n | lower | DECIMATE | 7 | +3.24e-05 | +3.15e-05 | -9.84e-06 | +8.36e-05 | 0.9531 | 1 | 1 | 5 | 0.156 | — | -0.643 | 0 | exploratory |
| 5% | roi_hausdorff_n | lower | QEM | 7 | +0.000266 | -1.17e-05 | -0.000237 | +0.0009 | 0.9531 | 4 | 0 | 3 | 1 | — | +0 | 0 | exploratory |
| 5% | roi_hausdorff_n | lower | DECIMATE | 7 | +0.000467 | +0.000385 | -0.00018 | +0.00122 | 0.9531 | 3 | 0 | 4 | 0.219 | — | -0.571 | 0 | exploratory |
| 5% | roi_chamfer_sym_n | lower | QEM | 7 | +2.52e-05 | +1.62e-05 | -2.51e-05 | +8.64e-05 | 0.9531 | 3 | 0 | 4 | 0.469 | 0.469 | -0.357 | 0 | holm |
| 5% | roi_chamfer_sym_n | lower | DECIMATE | 7 | +3.17e-05 | +2.51e-05 | -1.94e-05 | +9.11e-05 | 0.9531 | 2 | 1 | 4 | 0.375 | — | -0.429 | 0 | exploratory |
| 5% | roi_hausdorff_sym_n | lower | QEM | 7 | +0.000239 | -1.17e-05 | -0.000237 | +0.000883 | 0.9531 | 4 | 0 | 3 | 1 | — | +0 | 0 | exploratory |
| 5% | roi_hausdorff_sym_n | lower | DECIMATE | 7 | +0.000467 | +0.000385 | -0.00018 | +0.00122 | 0.9531 | 3 | 0 | 4 | 0.219 | — | -0.571 | 0 | exploratory |
| 5% | nonroi_chamfer_n | lower | QEM | 7 | +6.47e-06 | +8.74e-06 | -1.4e-05 | +2.31e-05 | 0.9531 | 2 | 1 | 4 | 0.469 | — | -0.357 | 0 | exploratory |
| 5% | nonroi_chamfer_n | lower | DECIMATE | 7 | +8.17e-06 | +1.16e-05 | -1.42e-05 | +2.7e-05 | 0.9531 | 2 | 1 | 4 | 0.375 | — | -0.429 | 0 | exploratory |
| 5% | time_s | lower | QEM | 9 | +4.27 | +4.23 | +3.18 | +5.56 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |
| 5% | time_s | lower | DECIMATE | 9 | +13.3 | +14.4 | +8.38 | +17.7 | 0.9609 | 0 | 0 | 9 | 0.00391 | — | -1 | 0 | exploratory |

## 4. Per kategori (deskriptif; n per kategori kecil)

Sel = W/T/L PERCEPTUAL terhadap baseline (menang/seri/kalah, toleransi seperti di atas). Tabel lengkap dengan Δ, CI, p dan Holm per kategori: `tables/paired_by_category.md`.

| category | ratio | baseline | n (max) | Chamfer (norm.) (lower better) | Hausdorff sampled (norm.) (lower better) | Silhouette IoU, 13 views (higher better) | ROI Chamfer ref->LOD (norm.) (lower better) | ROI Chamfer symmetric (norm.) (lower better) |
|---|---:|---|---:|---|---|---|---|---|
| character_face | 50% | QEM | 3 | 2/0/1 | 0/0/3 | 0/3/0 | 0/0/3 | 0/0/3 |
| character_face | 50% | DECIMATE | 3 | 2/0/1 | 0/0/3 | 0/3/0 | 0/0/3 | 0/0/3 |
| character_face | 20% | QEM | 3 | 0/1/2 | 0/0/3 | 1/2/0 | 0/0/3 | 0/0/3 |
| character_face | 20% | DECIMATE | 3 | 0/2/1 | 0/0/3 | 1/2/0 | 0/0/3 | 0/0/3 |
| character_face | 5% | QEM | 3 | 2/0/1 | 0/0/3 | 1/2/0 | 0/0/3 | 0/0/3 |
| character_face | 5% | DECIMATE | 3 | 2/0/1 | 0/0/3 | 1/2/0 | 0/0/3 | 0/0/3 |
| hard_surface | 50% | QEM | 3 | 1/0/2 | 1/0/2 | 0/3/0 | 0/0/1 | 0/0/1 |
| hard_surface | 50% | DECIMATE | 3 | 1/0/2 | 1/0/2 | 0/3/0 | 0/0/1 | 0/0/1 |
| hard_surface | 20% | QEM | 3 | 0/0/3 | 0/0/3 | 0/3/0 | 0/0/1 | 1/0/0 |
| hard_surface | 20% | DECIMATE | 3 | 0/0/3 | 0/0/3 | 0/3/0 | 0/0/1 | 0/0/1 |
| hard_surface | 5% | QEM | 3 | 1/0/2 | 2/0/1 | 1/2/0 | 1/0/0 | 1/0/0 |
| hard_surface | 5% | DECIMATE | 3 | 0/0/3 | 2/0/1 | 0/2/1 | 0/0/1 | 0/1/0 |
| organic | 50% | QEM | 3 | 0/0/3 | 0/0/3 | 0/3/0 | 2/0/1 | 2/0/1 |
| organic | 50% | DECIMATE | 3 | 0/0/3 | 0/0/3 | 0/3/0 | 2/0/1 | 2/0/1 |
| organic | 20% | QEM | 3 | 0/0/3 | 0/0/3 | 0/3/0 | 2/0/1 | 2/0/1 |
| organic | 20% | DECIMATE | 3 | 0/0/3 | 0/0/3 | 0/3/0 | 2/0/1 | 2/0/1 |
| organic | 5% | QEM | 3 | 0/0/3 | 0/0/3 | 1/2/0 | 1/1/1 | 2/0/1 |
| organic | 5% | DECIMATE | 3 | 0/0/3 | 0/0/3 | 1/2/0 | 1/1/1 | 2/0/1 |

## 5. Ablation (Δ = varian − PERCEPTUAL penuh)

Menonaktifkan satu kanal menormalkan ulang bobot kanal aktif (rata-rata terbobot). ANCHOR_V01 memakai aturan anchor v0.1 (λ = 8) dengan importance v0.2, bukan metode v0.1 utuh. LAMBDA0 adalah kontrol (harus identik dengan QEM). Sel = rata-rata Δ antar-aset (jumlah aset varian lebih baik : lebih buruk). Tabel lengkap: `tables/ablation.md`.

| split | condition | ratio | n (max) | Chamfer (norm.) (lower better) | Hausdorff sampled (norm.) (lower better) | Silhouette IoU, 13 views (higher better) | ROI Chamfer ref->LOD (norm.) (lower better) | ROI Chamfer symmetric (norm.) (lower better) |
|---|---|---:|---:|---|---|---|---|---|
| evaluation | ANCHOR_V01 | 50% | 9 | +2.65e-05 (0:9) | +0.000708 (0:9) | -0.000435 (0:4) | +4.25e-05 (0:7) | +4.17e-05 (0:7) |
| evaluation | ANCHOR_V01 | 20% | 9 | +9.92e-05 (0:9) | +0.00153 (0:9) | -0.00174 (0:8) | +0.000132 (0:7) | +0.000129 (0:7) |
| evaluation | ANCHOR_V01 | 5% | 9 | +0.000423 (0:9) | +0.00366 (0:9) | -0.00832 (0:9) | +0.000477 (0:7) | +0.000467 (0:7) |
| evaluation | LAMBDA0 | 50% | 9 | -2.67e-06 (6:3) | -0.000104 (8:1) | -7.8e-05 (0:0) | -3.53e-06 (5:2) | -3.52e-06 (5:2) |
| evaluation | LAMBDA0 | 20% | 9 | -5.65e-06 (8:0) | -0.000324 (9:0) | -0.000193 (0:1) | -9.13e-06 (5:2) | -9.27e-06 (4:3) |
| evaluation | LAMBDA0 | 5% | 9 | -1.52e-05 (6:3) | -0.000248 (7:2) | -0.000529 (0:3) | -2.95e-05 (4:2) | -2.52e-05 (4:3) |
| evaluation | NO_SALIENCY | 50% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | NO_SALIENCY | 20% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | NO_SALIENCY | 5% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | NO_SHARP | 50% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | NO_SHARP | 20% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | NO_SHARP | 5% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | NO_SILHOUETTE | 50% | 9 | -2.67e-06 (6:3) | -0.000104 (8:1) | -7.8e-05 (0:0) | -3.53e-06 (5:2) | -3.52e-06 (5:2) |
| evaluation | NO_SILHOUETTE | 20% | 9 | -5.65e-06 (8:0) | -0.000324 (9:0) | -0.000193 (0:1) | -9.13e-06 (5:2) | -9.27e-06 (4:3) |
| evaluation | NO_SILHOUETTE | 5% | 9 | -1.52e-05 (6:3) | -0.000248 (7:2) | -0.000529 (0:3) | -2.95e-05 (4:2) | -2.52e-05 (4:3) |
| evaluation | ONLY_SALIENCY | 50% | 9 | -5.09e-08 (4:4) | -7.66e-05 (7:1) | -6.41e-05 (0:0) | -4.21e-06 (5:2) | -4.29e-06 (5:0) |
| evaluation | ONLY_SALIENCY | 20% | 9 | +2.27e-06 (3:3) | -0.000249 (9:0) | -0.000214 (0:1) | -7.85e-06 (4:3) | -8.17e-06 (4:3) |
| evaluation | ONLY_SALIENCY | 5% | 9 | +1.03e-05 (3:6) | +6.91e-05 (6:3) | -0.000646 (0:2) | -2.54e-05 (4:3) | -2.71e-05 (4:3) |
| evaluation | ONLY_SILHOUETTE | 50% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | ONLY_SILHOUETTE | 20% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |
| evaluation | ONLY_SILHOUETTE | 5% | 9 | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) | +0 (0:0) |

## 6. Sapuan λ

Nilai relatif = rata-rata antar-aset dari perubahan per aset terhadap QEM (λ = 0): persen untuk jarak, selisih absolut untuk IoU. Pareto dihitung hanya di antara nilai λ sapuan dan hanya pada split `tuning`, satu-satunya split tempat λ boleh dipilih.

Split `evaluation` (eksploratori; bukan dasar memilih λ, tanpa penanda Pareto):

| ratio | condition | λ | n | Chamfer vs QEM | Hausdorff vs QEM | Δ IoU axis | Δ IoU 13v | ROI Chamfer vs QEM | ROI Chamfer sym. vs QEM | Pareto IoU/H | Pareto ROI/global |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | sweep/LAMBDA_0 | 0 | 9 | +0.00% | +0.00% | +0 | +0 | +0.00% | +0.00% | — | — |
| 50% | sweep/LAMBDA_1 | 1 | 9 | +4.11% | +30.62% | +1.2e-05 | +2.9e-05 | +1.38% | +1.48% | — | — |
| 50% | sweep/LAMBDA_2 | 2 | 9 | -0.02% | +3.32% | +2.65e-05 | +4.42e-05 | +3.00% | +3.36% | — | — |
| 50% | sweep/LAMBDA_3 | 3 | 9 | +8.54% | +95.87% | +2.5e-05 | +5.01e-05 | +4.01% | +3.81% | — | — |
| 50% | sweep/LAMBDA_5 | 5 | 9 | +1.81% | +39.56% | +3.95e-05 | +6.1e-05 | +6.18% | +5.57% | — | — |
| 50% | sweep/LAMBDA_8 | 8 | 9 | +5.79% | +124.12% | +4.64e-05 | +7.02e-05 | +9.16% | +6.99% | — | — |
| 50% | sweep/LAMBDA_12 | 12 | 9 | +3.63% | +39.73% | +5.45e-05 | +7.8e-05 | +11.10% | +11.30% | — | — |
| 50% | sweep/LAMBDA_20 | 20 | 9 | +5.34% | +52.37% | +5.54e-05 | +8.27e-05 | +14.20% | +11.79% | — | — |
| 50% | main/PERCEPTUAL | 12 | 9 | +3.63% | +39.73% | +5.45e-05 | +7.8e-05 | +11.10% | +11.30% | — | — |
| 50% | main/DECIMATE | — | 9 | +2.56% | +95.17% | +9.95e-07 | -2.79e-07 | -0.17% | -0.58% | — | — |
| 50% | ablation/ANCHOR_V01 | 8 | 9 | +150.43% | +495.65% | -0.00031 | -0.000357 | +203.25% | +195.24% | — | — |
| 20% | sweep/LAMBDA_0 | 0 | 9 | +0.00% | +0.00% | +0 | +0 | +0.00% | +0.00% | — | — |
| 20% | sweep/LAMBDA_1 | 1 | 9 | +0.32% | +9.00% | +9.96e-05 | +9.07e-05 | +1.19% | +0.68% | — | — |
| 20% | sweep/LAMBDA_2 | 2 | 9 | +1.28% | +9.71% | +0.00012 | +0.000114 | +2.90% | +2.91% | — | — |
| 20% | sweep/LAMBDA_3 | 3 | 9 | +2.31% | +12.97% | +0.000129 | +0.000133 | +3.85% | +2.61% | — | — |
| 20% | sweep/LAMBDA_5 | 5 | 9 | +2.66% | +17.68% | +0.00014 | +0.000154 | +6.02% | +5.94% | — | — |
| 20% | sweep/LAMBDA_8 | 8 | 9 | +4.09% | +36.16% | +0.000147 | +0.000172 | +7.44% | +7.66% | — | — |
| 20% | sweep/LAMBDA_12 | 12 | 9 | +6.12% | +125.50% | +0.000166 | +0.000193 | +8.65% | +8.79% | — | — |
| 20% | sweep/LAMBDA_20 | 20 | 9 | +6.48% | +67.65% | +0.0002 | +0.000207 | +12.78% | +13.69% | — | — |
| 20% | main/PERCEPTUAL | 12 | 9 | +6.12% | +125.50% | +0.000166 | +0.000193 | +8.65% | +8.79% | — | — |
| 20% | main/DECIMATE | — | 9 | -0.09% | +4.47% | +2.37e-05 | +3.1e-05 | -0.18% | +0.17% | — | — |
| 20% | ablation/ANCHOR_V01 | 8 | 9 | +332.91% | +708.78% | -0.00147 | -0.00154 | +187.02% | +186.80% | — | — |
| 5% | sweep/LAMBDA_0 | 0 | 9 | +0.00% | +0.00% | +0 | +0 | +0.00% | +0.00% | — | — |
| 5% | sweep/LAMBDA_1 | 1 | 9 | -0.06% | +0.20% | +8.94e-05 | +0.000244 | +3.31% | +2.34% | — | — |
| 5% | sweep/LAMBDA_2 | 2 | 9 | -0.21% | +1.45% | +8.35e-05 | +0.000303 | +4.80% | +2.91% | — | — |
| 5% | sweep/LAMBDA_3 | 3 | 9 | +0.20% | +2.98% | +0.000224 | +0.000403 | +5.39% | +4.34% | — | — |
| 5% | sweep/LAMBDA_5 | 5 | 9 | +0.63% | +6.39% | +0.000256 | +0.000431 | +4.53% | +3.35% | — | — |
| 5% | sweep/LAMBDA_8 | 8 | 9 | +1.84% | +23.16% | +0.000302 | +0.000472 | +7.42% | +4.54% | — | — |
| 5% | sweep/LAMBDA_12 | 12 | 9 | +2.37% | +30.06% | +0.000249 | +0.000529 | +6.36% | +4.68% | — | — |
| 5% | sweep/LAMBDA_20 | 20 | 9 | +3.60% | +40.71% | +0.000337 | +0.00057 | +7.65% | +5.15% | — | — |
| 5% | main/PERCEPTUAL | 12 | 9 | +2.37% | +30.06% | +0.000249 | +0.000529 | +6.36% | +4.68% | — | — |
| 5% | main/DECIMATE | — | 9 | -1.37% | -0.25% | +0.000474 | +0.000424 | -1.25% | -2.37% | — | — |
| 5% | ablation/ANCHOR_V01 | 8 | 9 | +200.63% | +271.64% | -0.00808 | -0.0078 | +177.97% | +171.88% | — | — |

## 7. Waktu (median antar-pengulangan, tanpa warm-up)

Per LOD = biaya mandiri membuat LOD itu: DECIMATE satu run modifier Triangulate + Decimate dan `new_from_object` (pembacaan LOD ke larik NumPy hanya untuk pengukuran, tidak termasuk; median set di kolom "Decimate readback"); QEM/PERCEPTUAL konversi mesh sumber ke larik + kunci seam/material (`feature_constraints`) + waktu pass kumulatif sampai snapshot + rekonstruksi mesh (+ peta kepentingan untuk PERCEPTUAL). Kolom "incl." adalah median komponen yang sudah termasuk. "all LODs" = seluruh set LOD satu metode. Metrik tidak termasuk. Record timing lama (tanpa kolom komponen) memakai definisi sebelumnya: DECIMATE termasuk pembacaan, QEM/PERCEPTUAL tanpa kunci.

| asset | split | method | source tris | all LODs median s | Q1 | Q3 | reps | incl. input arrays s | incl. seam/material locks s | Decimate readback s (excluded) | per-LOD median |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| beetle | evaluation | DECIMATE | 49,272 | 1.123 | 1.062 | 1.227 | 3 | — | — | 0.06346 | 50%:0.305s 20%:0.374s 5%:0.424s |
| beetle | evaluation | QEM | 49,272 | 7.515 | 7.463 | 8.077 | 3 | 0.07662 | 0.03878 | — | 50%:4.15s 20%:6.19s 5%:7.23s |
| beetle | evaluation | PERCEPTUAL | 49,272 | 12.19 | 11.71 | 12.19 | 3 | 0.07689 | 0.04499 | — | 50%:8.38s 20%:10.5s 5%:11.9s |
| fandisk | evaluation | DECIMATE | 12,946 | 0.1864 | 0.1739 | 0.194 | 3 | — | — | 0.01396 | 50%:0.0476s 20%:0.0683s 5%:0.0705s |
| fandisk | evaluation | QEM | 12,946 | 1.86 | 1.718 | 1.861 | 3 | 0.02087 | 0.01064 | — | 50%:1.02s 20%:1.48s 5%:1.78s |
| fandisk | evaluation | PERCEPTUAL | 12,946 | 4.085 | 4.036 | 4.138 | 3 | 0.01801 | 0.0108 | — | 50%:3.23s 20%:3.75s 5%:4.01s |
| rocker_arm | evaluation | DECIMATE | 20,088 | 0.2846 | 0.2834 | 0.2864 | 3 | — | — | 0.01805 | 50%:0.0742s 20%:0.0975s 5%:0.117s |
| rocker_arm | evaluation | QEM | 20,088 | 2.597 | 2.558 | 2.707 | 3 | 0.02559 | 0.01313 | — | 50%:1.47s 20%:2.17s 5%:2.51s |
| rocker_arm | evaluation | PERCEPTUAL | 20,088 | 4.734 | 4.698 | 4.785 | 3 | 0.02324 | 0.01377 | — | 50%:3.66s 20%:4.28s 5%:4.64s |
| max_planck | evaluation | DECIMATE | 99,991 | 2.261 | 2.228 | 2.322 | 3 | — | — | 0.08232 | 50%:0.59s 20%:0.763s 5%:0.886s |
| max_planck | evaluation | QEM | 99,991 | 13.46 | 13.4 | 13.61 | 3 | 0.1221 | 0.07112 | — | 50%:7.34s 20%:11.1s 5%:13s |
| max_planck | evaluation | PERCEPTUAL | 99,991 | 19.03 | 18.73 | 19.4 | 3 | 0.124 | 0.07041 | — | 50%:13s 20%:16.7s 5%:18.6s |
| igea | evaluation | DECIMATE | 100,000 | 2.144 | 2.097 | 2.357 | 3 | — | — | 0.08851 | 50%:0.535s 20%:0.726s 5%:0.856s |
| igea | evaluation | QEM | 100,000 | 13.12 | 12.91 | 13.29 | 3 | 0.1153 | 0.06619 | — | 50%:7.26s 20%:10.7s 5%:12.6s |
| igea | evaluation | PERCEPTUAL | 100,000 | 18.97 | 18.78 | 19.05 | 3 | 0.1162 | 0.04807 | — | 50%:13s 20%:16.7s 5%:18.5s |
| ogre | evaluation | DECIMATE | 100,000 | 2.444 | 2.375 | 2.445 | 3 | — | — | 0.09173 | 50%:0.589s 20%:0.85s 5%:0.917s |
| ogre | evaluation | QEM | 100,000 | 13.04 | 12.95 | 13.13 | 3 | 0.1174 | 0.06224 | — | 50%:7.35s 20%:10.8s 5%:12.6s |
| ogre | evaluation | PERCEPTUAL | 100,000 | 17.3 | 16.9 | 17.55 | 3 | 0.1217 | 0.05684 | — | 50%:11.6s 20%:15s 5%:16.8s |
| bunny | evaluation | DECIMATE | 69,451 | 1.494 | 1.49 | 1.554 | 3 | — | — | 0.05845 | 50%:0.397s 20%:0.522s 5%:0.578s |
| bunny | evaluation | QEM | 69,451 | 9.457 | 9.391 | 9.568 | 3 | 0.08431 | 0.04787 | — | 50%:5.4s 20%:7.96s 5%:9.16s |
| bunny | evaluation | PERCEPTUAL | 69,451 | 13.73 | 13.62 | 13.85 | 3 | 0.07552 | 0.04707 | — | 50%:9.64s 20%:12.1s 5%:13.4s |
| armadillo | evaluation | DECIMATE | 99,976 | 2.663 | 2.654 | 2.821 | 3 | — | — | 0.08614 | 50%:0.71s 20%:0.899s 5%:1.06s |
| armadillo | evaluation | QEM | 99,976 | 14.52 | 14.38 | 14.53 | 3 | 0.1171 | 0.06514 | — | 50%:8.07s 20%:11.9s 5%:13.9s |
| armadillo | evaluation | PERCEPTUAL | 99,976 | 20.03 | 19.88 | 20.07 | 3 | 0.126 | 0.0654 | — | 50%:13.3s 20%:17.2s 5%:19.5s |
| horse | evaluation | DECIMATE | 96,966 | 2.384 | 2.163 | 2.471 | 3 | — | — | 0.08718 | 50%:0.602s 20%:0.839s 5%:0.945s |
| horse | evaluation | QEM | 96,966 | 14.45 | 13.53 | 14.85 | 3 | 0.1155 | 0.07246 | — | 50%:8.32s 20%:12.1s 5%:13.9s |
| horse | evaluation | PERCEPTUAL | 96,966 | 18.26 | 15.86 | 18.35 | 3 | 0.1155 | 0.0522 | — | 50%:12.3s 20%:15.6s 5%:17.8s |

## 8. Gambar

- [Kualitas terhadap jumlah segitiga aktual, split evaluation](figures/quality_vs_triangles_evaluation.png)
- [Sapuan λ, split evaluation (eksploratori, tanpa penanda Pareto)](figures/lambda_pareto_evaluation.png)
- [Waktu pemrosesan terhadap jumlah segitiga sumber](figures/timing_vs_source_triangles.png)

## 9. Tabel

- `tables/all_quality_rows.md` (+ `.csv`)
- `tables/per_asset.md` (+ `.csv`)
- `tables/paired_main.md` (+ `.csv`)
- `tables/paired_by_category.md` (+ `.csv`)
- `tables/ablation.md` (+ `.csv`)
- `tables/lambda_sweep.md` (+ `.csv`)
- `tables/timing.md` (+ `.csv`)
