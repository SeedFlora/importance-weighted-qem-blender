# Rancangan dan metode penelitian Smart LOD (v0.2)

**Judul:** *Perceptual-Aware Automatic Level-of-Detail Generation for 3D Assets Using a Blender Extension*.

Dokumen ini menjelaskan metode yang diimplementasikan pada extension **v0.2.0** dan protokol penelitian yang benar-benar dijalankan. Ringkasan hasil ada di bagian 7, dan semua angkanya disalin dari berkas hasil di `results/`. Istilah *perceptual-aware* di sini berarti penggunaan indikator geometris dan visibilitas sebagai pendekatan terhadap kepentingan visual. Prototipe tidak mendeteksi wajah dan tidak menggunakan machine learning. Kesesuaian indikator tersebut dengan persepsi manusia juga belum dibuktikan, karena studi pengguna belum dilaksanakan.

## 1. Pertanyaan, hipotesis, dan kontribusi

Pertanyaan utama: pada jumlah segitiga hasil yang sama, apakah memperbesar kuadrik QEM di wilayah yang dinilai penting secara visual mempertahankan silhouette dan area penting lebih baik daripada QEM tanpa bobot? Pertanyaan berikutnya: berapa biayanya pada galat global dan waktu, bagaimana posisinya terhadap Blender Decimate, dan apakah perbedaannya terlihat oleh pengguna?

Hipotesis primer ditetapkan dalam [praregistrasi](../results/evaluation/PREREGISTRATION.md) setelah penalaan dan sebelum evaluasi. Keduanya membandingkan PERCEPTUAL dengan QEM pada rasio 0,05 (LOD3):

- **H1 (silhouette):** IoU silhouette 13 arah PERCEPTUAL lebih tinggi (n = 9 aset evaluasi).
- **H2 (area penting):** Chamfer ROI simetris PERCEPTUAL lebih rendah (n = 7 aset evaluasi yang memiliki ROI).

Keduanya diuji dengan Wilcoxon signed-rank eksak dua sisi, lalu dikoreksi Holm sebagai satu keluarga berisi dua uji pada α = 0,05. Praregistrasi juga mencatat prediksi sebelum data dilihat: H1 mungkin didukung, sedangkan H2 kemungkinan tidak. **H3** (penilaian pengguna pada kualitas keseluruhan, LOD2 dan LOD3) didefinisikan di Bab 3 skripsi, di luar praregistrasi dan di luar keluarga Holm primer. Datanya belum ada.

Hipotesis dapat ditolak, dan tidak dirumuskan sebagai jaminan bahwa semua metrik membaik. Proteksi satu wilayah dapat mengorbankan akurasi wilayah lain; kenaikan galat global karena itu dilaporkan sebagai biaya.

Kontribusi yang realistis untuk diuji adalah integrasi indikator kepentingan ke dalam QEM berbobot, extension Blender yang dapat direproduksi, dan evaluasi terkendali dengan pemisahan aset penalaan dan evaluasi. Pembobotan kuadrik dengan peta kepentingan bukan gagasan baru; Lee, Varshney & Jacobs (2005) sudah memakai saliency untuk membobot simplifikasi berbasis kuadrik. Klaim kebaruan karena itu memerlukan tinjauan pustaka (Bab 2 skripsi); implementasi fitur saja tidak membuktikannya.

## 2. Tiga metode pembanding

| Kode internal | Metode | Peran |
| --- | --- | --- |
| DECIMATE | Blender Decimate, mode Collapse (Triangulate + `use_collapse_triangulate`) | Pembanding praktis dari Blender |
| QEM | Implementasi edge-collapse QEM dalam proyek, tanpa bobot | Pembanding terkendali: engine, snapshot, dan pemeriksaan validitas sama |
| PERCEPTUAL | QEM yang sama dengan kuadrik berbobot `1 + λ·I` | Metode usulan |

QEM merangkum galat terhadap bidang permukaan menggunakan matriks kuadrik; biaya gabungan dipakai untuk memilih kontraksi mesh. Implementasi proyek membatasi kontraksi pada edge dan menambahkan pemeriksaan validitas, sehingga tidak boleh disebut reproduksi lengkap seluruh algoritma atau hasil publikasi asli. [Garland dan Heckbert, 1997](https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf).

QEM dan PERCEPTUAL hanya berbeda pada faktor bobot kuadrik. Selisih keduanya karena itu dapat diatribusikan pada pembobotan. Kesamaan ini diperiksa pada setiap run: PERCEPTUAL dengan λ = 0 (kondisi LAMBDA0) harus identik larik-per-larik dengan QEM.

Blender Decimate Collapse juga menggunakan kuadrik dan evaluasi biaya kontraksi pada implementasi resminya. Jadi, perbandingan ini bukan tiga keluarga algoritma yang sepenuhnya berbeda. Selisih terhadap DECIMATE mencerminkan implementasi C/C++ dan aturan validitas yang berbeda, sehingga dibaca sebagai gambaran manfaat praktis, bukan bukti mekanisme. Catat versi Blender karena implementasinya dapat berubah. [Sumber resmi Blender, `bmesh_decimate_collapse.cc`](https://github.com/blender/blender/blob/main/source/blender/bmesh/tools/bmesh_decimate_collapse.cc).

Rasio Decimate dihitung berdasarkan segitiga; polygon berupa quad atau ngon dapat membuat hitungan tampak berbeda jika triangulasi hasil tidak diaktifkan. Pipeline karena itu mentriangulasi input dan mengaktifkan triangulasi hasil, lalu melaporkan jumlah segitiga aktual. [Blender Manual: Decimate Modifier](https://docs.blender.org/manual/id/dev/modeling/modifiers/generate/decimate.html).

Opsi **Lock UV / material seams** (default aktif di UI) hanya berlaku untuk QEM dan PERCEPTUAL. Dengan kunci aktif, perbedaan terhadap Decimate juga mencakup kebijakan kendala, dan target rendah dapat tidak tercapai. Evaluasi penelitian karena itu menjalankan semua metode **tanpa** kunci seam (`--no-preserve-seams`), sehingga kendalanya setara dengan Decimate. Kondisi dengan seam terkunci, bila dijalankan, harus dilaporkan sebagai analisis terpisah.

## 3. Metode usulan

### 3.1 Snapshot dan normalisasi

Sumber diambil sebagai mesh terevaluasi pada frame aktif, termasuk modifier, dengan transformasi dunia di-bake. Transformasi cermin membalik urutan vertex, sehingga face dibalik kembali agar normal tetap menghadap ke luar. Loop triangle hasil snapshot dipakai bersama oleh ketiga metode, oleh metrik, dan oleh transfer UV.

Solver QEM menormalisasi posisi: pusat di titik tengah bounding box, lalu dibagi panjang sisi terpanjangnya. Metrik memakai normalisasi lain, yaitu diagonal bounding box referensi (bagian 5). Jangan mencampur keduanya.

### 3.2 Kanal kepentingan

Semua kanal dihitung per vertex pada mesh sumber sebelum penyederhanaan, bernilai dalam `[0, 1]`, dan murni geometris: tidak ada deteksi semantik. Kode ada di `smart_lod/perception.py`.

| Kanal | Definisi singkat | Bobot default |
| --- | --- | --- |
| `saliency` | Mesh saliency multi-skala (Lee, Varshney & Jacobs 2005), dengan penyimpangan yang dijelaskan di bawah | 0 |
| `silhouette` | Seberapa sering vertex berada pada atau dekat garis luar objek dari 64 arah ortografik generik | **1** |
| `sharp` | Edge tajam, tepi terbuka/non-manifold, dan edge *Mark Sharp*, disebar satu ring | 0 |
| `visibility` | Visibilitas dari kamera scene (frustum, menghadap kamera, tidak tertutup mesh sendiri) | 0,15, hanya bila kamera dipakai |
| `manual` | Bobot vertex group `LOD_Protect` yang dicat pengguna | digabung lewat maksimum |

**Silhouette.** Arah pandang adalah 64 titik Fibonacci pada belahan bola atas; arah `d` dan `−d` memberi silhouette cermin yang identik. Untuk setiap arah, gabungan semua segitiga diproyeksikan secara ortografik ke bingkai persegi (1,04 × rentang proyeksi) dan dirasterisasi pada 256 × 256 piksel. Piksel garis luar adalah piksel mask yang punya tetangga-4 di luar mask; tepi gambar dihitung sebagai luar. Pita garis luar dilebarkan `1 + k` piksel (jarak Chebyshev), dengan `k = ceil(resolusi/2 × (1 − cos ρ))`, yaitu 2 piksel pada 64 arah dan 256 px.

Bobot kontur tiap vertex dibuat halus. Misalkan `[lo, hi]` rentang `n · d` atas normal face insiden, dan `gap = max(lo, −hi, 0)`. Nilai `gap` nol bila face insiden menghadap ke dua sisi, yaitu vertex kontur. Bobotnya:

```text
w_v(d) = max(τ_v − gap, 0) / τ,    τ = sin ρ,  cos ρ = 1 − 1/64  (ρ ≈ 10,1°)
τ_v    = min(τ, sqrt(s · S · (1 − cos ρ)))
```

`s` adalah gradien permukaan `n · d` (kuadrat terkecil atas centroid face insiden) dan `S` sisi bingkai. Wilayah melengkung dan lipatan mendapat kernel penuh. Wilayah datar mendapat 0, karena bidang hanya berada di garis luar tepat saat dilihat dari samping. Vertex pada edge terbuka atau non-manifold mendapat bobot 1 di setiap arah.

Bobot hanya dihitung bila piksel vertex berada di dalam pita garis luar. Kontur yang tersembunyi di dalam mask proyeksi, misalnya rongga atau lipatan cekung, karena itu tidak dihitung, sedangkan lubang yang tembus pandang (torus) tetap dihitung. Jumlah atas semua arah dibagi persentil ke-90 dari jumlah positif, lalu dipotong ke 1.

Kernel halus yang disesuaikan dengan jarak antar-arah menggantikan uji kontur biner 24 arah pada versi awal v0.2. Uji biner itu membentuk pola garis yang bergantung pada orientasi dan tessellation mesh. Arah kanal ini generik dan bukan 13 arah metrik IoU. Keduanya tetap mengukur garis luar ortografik, dan kedekatan ini perlu diingat saat membaca H1.

**Saliency.** Kelengkungan rata-rata tak bertanda `|H|` dihitung dari Laplacian kotangen yang dijepit nonnegatif dengan luas barycentric (komponen normal saja), lalu dipotong pada persentil ke-99. Vertex pada tepi terbuka mengambil rata-rata tetangga interior, diisi ring demi ring, karena estimasi satu sisi di sana bias. Untuk `σ = kε`, `k ∈ {2, 3, 4, 5, 6}`, dan `ε = 0,003 × diagonal bounding box`, dihitung `S_k = |G(σ) − G(2σ)|`. Setiap `S_k` dinormalisasi ke maksimumnya dan dikalikan `(1 − m)²`, dengan `m` rata-rata maksimum lokal lainnya. Semua skala lalu dijumlahkan dan dibagi maksimum.

Penyimpangan dari makalah asli dinyatakan terbuka:

- `G(σ)` adalah satu langkah difusi panas implisit `(M + tL)u = M·H` dengan `t = σ²/2`, bukan rata-rata Gaussian Euklidean. Variansnya sama, tetapi bentuk kernel dan pengukuran jaraknya berbeda (intrinsik pada permukaan).
- Ada gerbang derau: bila kontras terkuat di bawah 0,02 × kelengkungan terpotong, hasil diskalakan `kontras / 0,02`. Tanpa gerbang, derau diskretisasi bola akan direntangkan menjadi `[0, 1]`. Nilai 0,02 dipilih pengembang dari beberapa mesh uji.
- Penguatan saliency pada makalah asli tidak dipakai, karena penguatan dilakukan seragam untuk semua kanal melalui λ.

**Sharp.** Sebuah edge tajam bila sudut antara dua normal face-nya ≥ `sharp_angle` (default 45°, orientasi face direkonsiliasi per edge) atau bila jumlah face-nya bukan dua. Vertex pada edge tajam dan vertex edge yang ditandai *Mark Sharp* di Blender bernilai 1, dan tetangga satu ring bernilai setengahnya.

**Visibility.** Vertex bernilai 1 bila memenuhi tiga syarat. Pertama, berada di dalam frustum kamera (sudut pandang atau kotak ortografik dan rentang clip). Kedua, menghadap kamera: ada face insiden dengan normal ke luar yang menghadap kamera, dan bila seluruh mesh berorientasi ke dalam (volume bertanda negatif), semua normal dibalik untuk uji ini. Ketiga, tidak tertutup mesh itu sendiri menurut ray BVH dari bidang clip dekat. Objek lain di scene tidak diperhitungkan. Lens shift diganti frustum simetris terkecil yang memuatnya, dan kamera panoramik diperlakukan sebagai perspektif. Kanal ini aktif hanya bila opsi kamera dinyalakan, kamera scene ada, dan bobotnya > 0.

**Kombinasi.**

```text
G_i = Σ[k aktif] w_k · f_k(i) / Σ[k aktif] w_k        (G_i = 0 bila tidak ada kanal aktif)
I_i = clamp(max(G_i, manual_i), 0, 1)
```

Kanal aktif hanya bila bobotnya > 0, dan `visibility` hanya bila ada kamera. Kanal nonaktif tidak ikut membagi. Mask manual tidak dirata-ratakan, sehingga tidak tereduksi oleh wilayah lain. Dengan bobot default dan tanpa kamera, `I_i = max(f_sil(i), manual_i)`. Dengan kamera, `G_i = (f_sil + 0,15 · f_vis) / 1,15`.

Default ini dipilih pada aset penalaan dengan aturan tertulis (bagian 4). Nilai sebelumnya tidak dipakai lagi: v0.1 memakai kelengkungan `θ/π`, silhouette arah sumbu tanpa uji oklusi, dan bobot 0,30 / 0,25 / 0,30 / 0,15, sedangkan versi awal v0.2 memakai 0,35 / 0,40 / 0,25 untuk saliency / silhouette / sharp. Sebagai gambaran biaya, kanal silhouette memakan 1,15 s pada mobil demo dengan 13.144 segitiga (`examples/demo_benchmark.json`).

Proteksi wajah memerlukan pengecatan grup oleh pengguna. Tanpa grup itu, sistem hanya melihat bentuk garis luar, bukan makna wajah. Eksperimen otomatis dan eksperimen berbantuan mask karena itu dipisahkan. Gunakan istilah “indikator keterpaparan terhadap arah pandang/kamera”, bukan “perhatian visual aktual”, kecuali ada data penggunaan yang mendukungnya.

### 3.3 Fungsi biaya: kuadrik berbobot (default)

Untuk face `f` dengan normal satuan `n_f` dan titik `x_f`, bidangnya `p_f = (n_f, −n_f · x_f)` dan kuadriknya `K_f = p_f p_fᵀ`, tanpa pembobotan luas. Setiap edge batas menambah bidang yang tegak lurus face-nya melalui edge itu, dengan bobot 10. Kuadrik vertex dan versi berbobotnya:

```text
Q_i  = Σ[f ∋ i] K_f + 10 · Σ[e ∈ batas(i)] K_e
Q̃_i = (1 + λ · I_i) · Q_i                         (λ ≥ 0; default 12)

biaya kontraksi (a, b):  Δ(v̄) = v̄ᵀ (Q̃_a + Q̃_b) v̄,   v̄ = (x, y, z, 1)
```

Bobot dikalikan setelah kuadrik bidang dan kuadrik batas dijumlahkan. Posisi baru dipilih dari kandidat berurutan: titik tengah, kedua ujung, minimum pada segmen edge, dan minimum tak terbatas bila sistemnya terkondisi baik (hanya edge interior). Kandidat yang lebih akhir harus lebih murah secara ketat. Bila satu ujung terkunci, posisi tetap di ujung itu. Seri biaya nol dipecah dengan panjang edge. Saat cluster digabung, kuadriknya dijumlahkan, sehingga bobot vertex asal terbawa sepanjang pass. Kepentingan hasil untuk heatmap diwariskan sebagai maksimum cluster dan tidak dipakai dalam biaya.

Dengan λ = 12, galat bidang di sekitar vertex berkepentingan penuh dihitung 13 kali. Edge di wilayah penting karena itu dikontraksi belakangan, dan posisi optimal ditarik ke bidang milik vertex penting.

Ada dua konsekuensi yang perlu dinyatakan:

- Bila λ = 0 atau semua `I_i = 0`, perkalian dilewati, sehingga hasilnya identik bit-per-bit dengan QEM, termasuk pemecahan seri. Ini diuji otomatis dan diperiksa pada setiap run evaluasi (54/54 baris LAMBDA0 identik).
- Bobot kuadrik hanya menskalakan galat bidang. Wilayah yang benar-benar planar mempunyai galat bidang nol berapa pun bobotnya, sehingga **tidak terlindungi**. Bobot hanya mengubah urutan dan posisi collapse di wilayah yang melengkung atau berlipat.

λ di UI dibatasi 0–100. Engine menerima paling besar `1e6`, batas konservatif karena aritmetika kuadrik baru meluap mulai sekitar `1e102`.

### 3.4 Aturan *anchor* v0.1 (hanya kondisi ablation)

v0.1 memakai penalti perpindahan posisi yang diakumulasi sebagai kuadrik titik:

```text
E_anchor(p) = E_QEM(p) + λ · Σ[i ∈ cluster(a) ∪ cluster(b)] I_i · ||p − x_i||²
```

`x_i` adalah posisi asal vertex `i`. Suku kedua diminimalkan oleh titik berat posisi asal vertex penting, yang pada permukaan cembung berada di sisi dalam permukaan. Akibatnya mesh menyusut. Aturan ini tersedia sebagai **Legacy v0.1 positional anchor** (`weighting='anchor'`) hanya untuk ablation. Hasilnya identik bit-per-bit dengan engine v0.1 selama penjaga `source_flip` (bagian 3.5) tidak menolak collapse; ini diuji terhadap salinan engine v0.1 yang dibekukan di `tests/reference/qem_v01.py`.

Penyusutan ini terukur:

- **Diagnosis v0.1 pada mobil demo di LOD3** (sebelum penjaga `source_flip` ada; tercatat di docstring `smart_lod/qem.py` dan Bab 3 skripsi): rasio volume 0,873 dibanding 0,899 pada QEM, dan IoU silhouette 0,936 dibanding 0,956.
- **Ablation evaluasi `ANCHOR_V01`** (λ = 8 dengan peta kepentingan v0.2; bukan metode v0.1 utuh). Pada 5%, rata-rata perubahan terhadap QEM adalah Chamfer +200,6%, Hausdorff +271,6%, dan IoU 13 arah −0,0078. Chamfer dan Hausdorff-nya lebih buruk daripada PERCEPTUAL pada 9 dari 9 aset di setiap rasio. Rasio volume pada 5% lebih kecil daripada QEM pada 4 dari 5 aset evaluasi tertutup: armadillo 0,982 vs 0,992, horse 0,988 vs 0,997, rocker_arm 0,985 vs 0,992, dan igea 0,997 vs 0,999. Pada fandisk keduanya 1,001. Sumber: `results/evaluation/analysis/report.md` bagian 5–6 dan `tables/all_quality_rows.csv`.

Karena itu v0.2 beralih ke kuadrik berbobot. Tidak seperti anchor, bobot kuadrik tidak melindungi wilayah planar, tetapi juga tidak menarik vertex ke dalam permukaan.

### 3.5 Pemeriksaan validitas dan kendala

- Vertex non-manifold dibekukan sejak awal: vertex pada edge dengan lebih dari dua face, vertex bow-tie, atau vertex yang link-nya bukan satu siklus atau satu lintasan.
- Setiap collapse diperiksa dengan kondisi link, duplikasi face, luas minimum, dan normal. Normal setiap face yang berubah boleh berputar paling banyak sekitar 84° per collapse terhadap normal saat itu.
- **Penjaga `source_flip`** (baru di v0.2): collapse ditolak bila ada face yang akan memiliki `dot ≤ 0` dengan normal segitiga asalnya. Tanpa penjaga ini, putaran kecil dapat menumpuk hingga face terlipat terhadap sumbernya. Hal ini paling sering terjadi di sekitar vertex seam yang terkunci, tetapi juga tanpa kunci. Jumlah penolakan dicatat di `stats['rejected_collapses']['source_flip']`. Penjaga ini berlaku sama untuk QEM dan PERCEPTUAL.
- Edge yang menghubungkan batas dengan interior tidak dapat dikontraksi. Edge batas dibatasi pada segmennya dan diberi kuadrik bidang batas.
- Segitiga terakhir setiap komponen permukaan dipertahankan, sehingga bagian kecil yang terpisah tidak hilang seluruhnya.
- Vertex terkunci (seam/material) mempertahankan posisi tepatnya. Edge dengan dua ujung terkunci tidak dikontraksi.

Akibatnya target jumlah segitiga tidak selalu tercapai. Satu collapse menghapus satu atau dua segitiga, sehingga pass dapat berhenti satu segitiga di bawah target. Selalu laporkan jumlah aktual dan `stop_reason`. Perlindungan lokal ini tidak membuktikan bahwa seluruh self-intersection global dapat dicegah.

### 3.6 Satu pass untuk banyak LOD

Urutan collapse greedy tidak bergantung pada anggaran; anggaran hanya menentukan kapan berhenti. `simplify_levels` karena itu menjalankan satu pass sampai anggaran terkecil dan mengambil snapshot saat setiap anggaran pertama kali tercapai. Setiap snapshot identik larik-per-larik dengan run `simplify` terpisah ke anggaran itu, termasuk statistiknya kecuali `seconds`. Nilai `seconds` adalah waktu kumulatif dari awal pass sampai snapshot. Satu set LOD karena itu kira-kira sama mahalnya dengan LOD terkecil saja.

## 4. Dataset dan protokol

Dataset berisi 12 aset dalam tiga kategori: `hard_surface`, `character_face`, dan `organic` ([dataset/README.md](../dataset/README.md)). Tiga aset penalaan (car, suzanne, spot) hanya dipakai untuk memilih λ dan bobot. Sembilan aset evaluasi (tiga per kategori) tidak dipakai untuk memilih parameter apa pun. Tujuh aset evaluasi memiliki grup `LOD_EvalROI`; fandisk dan rocker_arm tidak. ROI didefinisikan dengan aturan geometris sederhana oleh satu orang dan hanya dipakai untuk evaluasi, tidak pernah untuk proteksi. Jangan menganggap banyak LOD dari satu aset sebagai banyak aset independen. Sumber, lisensi, jumlah segitiga, komponen, UV, dan kondisi topologi setiap aset dicatat di `dataset/manifest.json`.

1. **Pisahkan penalaan dan evaluasi.** Protokol penalaan ditulis sebelum run penalaan ([results/tuning/PROTOCOL.md](../results/tuning/PROTOCOL.md)). Grid-nya lima set bobot (W1 0,35/0,40/0,25; W2 silhouette saja; W3 saliency saja; W4 0,2/0,6/0,2; W5 0,5/0,5/0) × λ ∈ {0, 1, 2, 3, 5, 8, 12, 20}. Aturan seleksinya memaksimalkan rata-rata ΔIoU 13 arah (PERCEPTUAL − QEM) pada rasio 0,2 dan 0,05, dengan syarat rata-rata perubahan Chamfer global ≤ +5% pada kedua rasio. Seleksi dijalankan secara mekanis dan memilih **W2 dengan λ = 12** (`results/tuning/selection.txt`).
2. **Rekam lingkungan.** Catat versi Blender dan build hash, versi extension, hash SHA-256 setiap berkas `smart_lod/*.py` dan harness, OS, CPU, dan RAM (`environment.json` setiap run). Gunakan snapshot statis yang sama untuk seluruh metode.
3. **Rasio.** LOD0 `1,00`, LOD1 `0,50`, LOD2 `0,20`, dan LOD3 `0,05`, dengan target `max(4, round(T_sumber × rasio))`. Setiap LOD setara dengan run dari sumber (bagian 3.6), bukan proses bertingkat dari LOD sebelumnya.
4. **Konfigurasi beku.** Konfigurasi yang sama dipakai untuk semua aset evaluasi: silhouette saja, λ = 12, kuadrik berbobot, 64 arah, dan 256 px. Evaluasi tidak memakai mask manual maupun kamera, sehingga kanal visibility tidak aktif. Semua metode dijalankan tanpa kunci seam. λ dan bobot diberikan eksplisit di argumen agar tercatat.
5. **Jalankan dan ukur.** Semua metode dijalankan pada semua aset dan rasio dengan metrik 4.000 sampel per mesh, raster 512 px, dan seed 20260921. Waktu diukur dalam run terpisah yang berurutan, tanpa proses lain: satu warm-up dan tiga pengulangan, dengan urutan metode diacak per pengulangan. Tombol **Compare All 3 Methods** di UI memakai urutan tetap, sehingga tidak dipakai untuk pengukuran waktu penelitian.
6. **Laporkan target dan hasil aktual.** Bila selisih jumlah segitiga antar-metode lebih dari 2%, bandingkan pada kurva kualitas–segitiga. Jangan menafsirkan hasil yang lebih padat sebagai bukti metode lebih baik. Pada evaluasi, ke-486 baris kualitas mencapai targetnya, dan tidak ada pasangan metode utama dengan selisih > 2%.
7. **Simpan semuanya.** Simpan sumber, LOD (`results/evaluation/lods_all/<aset>.blend`), `raw.jsonl`, konfigurasi, dan gambar. Laporkan kegagalan dan target yang tidak tercapai, bukan hanya hasil yang bagus.

Setiap run evaluasi memeriksa integritasnya sendiri. Hasil pemeriksaan: pass QEM yang diulang identik pada 9/9 aset; λ = 0 identik dengan QEM pada 54/54 baris; hash snapshot dan jumlah vertex ROI sesuai manifest pada 9/9 aset; dan run waktu mereproduksi larik run kualitas pada 108/108 baris. Tidak ada record error.

**Praregistrasi dan amandemen.** Run penalaan pertama memilih W4 dengan λ = 2. Hasil itu dibatalkan sebelum evaluasi apa pun dijalankan, karena tinjauan kode adversarial (37 temuan terverifikasi) mengubah metode dan metrik. Perubahannya: kanal silhouette halus 64 arah, penjaga `source_flip`, bingkai IoU berbasis referensi, ROI dua arah, dan saliency di tepi terbuka. Penalaan diulang dengan grid dan aturan yang sama; satu-satunya perubahan protokol adalah raster metrik 512 px. Amandemen 1, yang dicatat sebelum evaluasi, juga mengganti metrik H2 menjadi Chamfer ROI simetris, menyatakan keluarga Holm secara eksplisit, dan memakai interval eksak Hodges–Lehmann. Perintah run lengkap ada di `results/evaluation/run_all.sh`.

**Ablation dan sapuan λ (eksploratori).** Kondisi ablation adalah `NO_SALIENCY`, `NO_SILHOUETTE`, `NO_SHARP`, `ONLY_SALIENCY`, `ONLY_SILHOUETTE`, `ANCHOR_V01`, dan kontrol `LAMBDA0`. Sapuan memakai λ ∈ {0, 1, 2, 3, 5, 8, 12, 20} dengan bobot akhir. Karena konfigurasi akhir memakai silhouette saja, `NO_SALIENCY`, `NO_SHARP`, dan `ONLY_SILHOUETTE` identik dengan PERCEPTUAL, sedangkan `NO_SILHOUETTE` sama dengan λ = 0. Mematikan satu kanal menormalkan ulang bobot kanal aktif; aturan ini perlu disebut saat menafsirkan ablation. Run tambahan dengan bobot lama W1 disimpan di `results/evaluation/explore_W1/` dan tidak dipakai untuk memilih apa pun.

**Eksperimen sekunder bantuan mask** (deskriptif): pada tiga aset `character_face`, PERCEPTUAL dijalankan dengan mask manual = `LOD_EvalROI` untuk mensimulasikan pengguna yang mengecat wajah (`scripts/run_mask_experiment.py`). Karena area yang dilindungi sama dengan area yang diukur, desain ini sirkular. Tujuannya hanya menunjukkan bahwa kontrol manual bekerja dan berapa biayanya di luar area tersebut.

**Catatan kontaminasi.** `bunny` (aset evaluasi) sempat dipakai selama pengembangan untuk profil kecepatan dan pemeriksaan visual heatmap saliency, tetapi tidak untuk memilih λ atau bobot.

## 5. Definisi evaluasi

Implementasi ada di `smart_lod/metrics.py`. Semua ukuran dihitung antara snapshot sumber (referensi) dan LOD dalam koordinat dunia.

### Reduksi segitiga

```text
reduction = 1 − T_LOD / T_sumber
```

Hitung segitiga sesudah triangulasi, bukan jumlah polygon mesh campuran. LOD0 adalah referensi tanpa reduksi dan tidak menjadi bukti performa algoritma penyederhanaan.

### Jarak permukaan

Koordinat dinormalisasi sebelum operasi BVH float32 Blender: pusat di tengah bounding box referensi, lalu dibagi diagonalnya. Misalkan permukaan referensi `S`, hasil `L`, sampel permukaan referensi `P`, dan sampel hasil `R`. `d(p, L)` adalah jarak Euclidean ke titik terdekat pada permukaan segitiga `L`, bukan hanya ke vertex atau sampel lain.

```text
Chamfer(S, L)             = 0,5 × (mean[p ∈ P] d(p, L) + mean[r ∈ R] d(r, S))
Hausdorff_sampled(S, L)   = max(max[p ∈ P] d(p, L), max[r ∈ R] d(r, S))
Hausdorff_with_vertices   = max(Hausdorff_sampled, max[v ∈ V_S] d(v, L), max[v ∈ V_L] d(v, S))
```

Chamfer di sini adalah rata-rata simetris **tanpa kuadrat** dengan faktor `0,5`. Konvensi lain juga ada, sehingga definisinya wajib ikut laporan.

Sampel diambil seragam terhadap luas permukaan (koordinat barycentric akar kuadrat). Setiap mesh mendapat `sample_count` sampel dengan aliran acak terpisah dari `SeedSequence(seed)`: referensi, kandidat, ROI, dan non-ROI.

`Hausdorff_sampled` adalah estimasi dan dapat melewatkan ujung spike atau segitiga kecil yang jauh. `Hausdorff_with_vertices` juga mengukur setiap vertex segitiga nondegenerat kedua mesh ke permukaan lawannya. Nilainya masih batas bawah dari Hausdorff eksak, karena titik terjauh dapat terletak di dalam face.

Segitiga berluas nol dikeluarkan dari sampling, BVH, dan rasterisasi, tetapi tetap dihitung dalam jumlah segitiga. Versi `*_normalized` dibagi diagonal bounding box referensi, sehingga dapat dibandingkan antar-skala. Nilai lebih kecil lebih baik. Mesh yang sepenuhnya degenerat menghasilkan galat, bukan skor nol.

### Jarak di ROI

ROI adalah segitiga referensi nondegenerat yang rata-rata bobot `LOD_EvalROI` pada sudutnya ≥ 0,5; sisanya disebut non-ROI. Untuk setiap wilayah:

- `*_ref_to_candidate`: `sample_count` sampel pada wilayah referensi itu, diukur ke permukaan LOD.
- `*_candidate_to_ref`: sampel global LOD yang segitiga referensi terdekatnya berada di wilayah itu. Korespondensinya sama dengan yang dipakai Chamfer global, sehingga artefak di sisi LOD, seperti spike di dalam ROI, ikut terlihat.
- `*_symmetric`: rerata dua rata-rata terarah (Chamfer) atau maksimum dua maksimum terarah (Hausdorff).

Wilayah kosong atau tanpa sampel LOD melaporkan `None`, bukan nol. Versi v0.2 awal hanya mengukur arah referensi → LOD, sehingga tidak melihat geometri berlebih di ROI; karena itu H2 memakai versi simetris.

### Kesamaan silhouette

Sumber dan hasil diproyeksikan secara ortografik, lalu dirasterisasi sebagai mask biner dengan uji pusat piksel:

```text
IoU(view) = area(mask_S ∩ mask_L) / area(mask_S ∪ mask_L)
```

- **6 sumbu** (`silhouette_iou_mean`): `+X`, `−X`, `+Y`, `−Y`, `+Z`, `−Z`. Pasangan arah berlawanan memberi mask cermin yang identik, sehingga rata-rata enam arah setara dengan rata-rata tiga proyeksi.
- **13 arah** (`silhouette_iou_multiview_mean`): 3 sumbu, 4 diagonal ruang, dan 6 diagonal bidang dari lingkungan 3 × 3 × 3. Setiap arah juga mewakili arah lawannya. Ini tetap himpunan arah tetap yang kecil, bukan integral atas semua arah pandang.

Bingkai setiap view ditentukan oleh referensi saja: persegi berukuran 1,04 × rentang proyeksi referensi yang lebih besar, berpusat pada referensi. Mask referensi karena itu identik untuk semua kandidat. Bagian kandidat yang keluar bingkai menambah kanvas dengan kisi piksel yang sama, paling banyak `resolution` piksel per sumbu. View yang melampaui batas itu dicatat di `*_frame_clipped_views`, dan IoU-nya menjadi batas atas. Versi v0.1 dan versi awal v0.2 membentuk bingkai dari gabungan batas referensi dan kandidat, sehingga ukuran piksel berubah untuk setiap kandidat.

Nilai lebih besar lebih baik, maksimum `1`. Jika kedua mask kosong, konvensinya IoU = 1. Default UI adalah raster 128 px. Evaluasi penelitian memakai 512 px, karena pada 128 px noise aliasing (sekitar ±2·10⁻⁴) sebesar efek yang dicari. IoU hanya mengukur bentuk proyeksi ortografik. Warna, tekstur, detail interior, shading normal, dan pandangan perspektif tidak tercakup.

### Rasio volume (hanya harness)

Harness penelitian mencatat `volume_ratio = V_bertanda(LOD) / V_bertanda(referensi)` hanya bila referensi tertutup, yaitu setiap edge tepat memiliki dua face. Dari sembilan aset evaluasi, lima tertutup: armadillo, fandisk, horse, igea, dan rocker_arm.

### Waktu pemrosesan

Implementasi memakai `time.perf_counter()`. Field laporan extension (`smart_lod/pipeline.py`) per baris LOD:

| Field | Cakupan |
| --- | --- |
| `simplification_and_mesh_seconds` | QEM/PERCEPTUAL: waktu pass **kumulatif** dari awal pass sampai snapshot LOD itu, setara biaya run mandiri ke anggaran itu, ditambah rekonstruksi mesh LOD itu. DECIMATE: satu run modifier mandiri + konversi mesh |
| `mesh_reconstruction_seconds` | Rekonstruksi mesh dan transfer UV/material LOD itu; `null` pada DECIMATE karena tidak terpisahkan dari modifier |
| `constraint_preparation_seconds` | Ekstraksi kunci seam/material dan edge sharp, yang diperlukan QEM/PERCEPTUAL; 0 pada DECIMATE |
| `importance_seconds` | Peta kepentingan (`importance_preparation_seconds`, termasuk membaca grup proteksi); hanya pada baris PERCEPTUAL |
| `processing_seconds` | `simplification_and_mesh_seconds + constraint_preparation_seconds + importance_seconds`: biaya mandiri membuat LOD itu |
| `evaluation_seconds` | Komputasi metrik baris itu, hanya bila evaluasi aktif |

Field tingkat laporan:

| Field | Cakupan |
| --- | --- |
| `method_pass_seconds` | Per metode: seluruh pass + rekonstruksi semua level + `constraint_preparation_seconds`, tanpa peta kepentingan. DECIMATE: jumlah tiga run |
| `importance_preparation_seconds` / `importance_channel_seconds` | Peta kepentingan sekali per run, total dan per kanal (termasuk topologi) |
| `source_preparation_seconds` | Snapshot bersama; tidak termasuk dalam field lain mana pun |
| `total_wall_seconds` | Run sampai laporan terbentuk, termasuk objek hasil dan evaluasi; tanpa serialisasi, text block, atau export |

Peta kepentingan tetap dihitung sekali meskipun hanya baseline yang dibuat, karena juga dipakai untuk heatmap. Biayanya hanya dibebankan ke PERCEPTUAL. **Jangan menjumlahkan** `processing_seconds` atau `simplification_and_mesh_seconds` tiga LOD QEM/PERCEPTUAL. Nilainya kumulatif, sehingga bagian awal pass terhitung sampai tiga kali. Biaya satu set LOD tanpa evaluasi adalah `method_pass_seconds[metode]`, ditambah `importance_preparation_seconds` untuk PERCEPTUAL. Snapshot sumber tidak termasuk dalam angka itu. Untuk DECIMATE, jumlah ketiga baris sama dengan `method_pass_seconds`.

Harness penelitian memakai definisi skema 2 yang setara antar-metode. Untuk QEM/PERCEPTUAL, waktunya adalah konversi mesh sumber ke larik + kunci seam/material + pass kumulatif sampai snapshot + rekonstruksi, ditambah peta kepentingan untuk PERCEPTUAL. Untuk DECIMATE, waktunya satu run objek sementara + modifier + `new_from_object`, tanpa pembacaan hasil ke larik. Rinciannya ada di [results/README.md](../results/README.md). Laporkan median dan rentang antar-kuartil pengulangan. Jangan memasukkan waktu metrik hanya ke salah satu metode. Waktu QEM Python dibanding Decimate C/C++ mencerminkan implementasi sekaligus metode, sehingga tidak boleh dipakai untuk menyimpulkan kompleksitas algoritmik.

### Penilaian pengguna

Gunakan [USER_STUDY.csv](USER_STUDY.csv) sebagai format data dan alat di [study/](../study/README.md) untuk membuat stimulus, formulir, dan analisis. Rekrut peserta dengan persetujuan sesuai ketentuan kampus, dan simpan ID anonim, bukan nama atau kontak, pada data penilaian. Rancangan awal 12–20 peserta dapat dipakai untuk pilot, tetapi bukan jaminan daya statistik yang memadai.

Tampilkan sumber bersama hasil dengan kamera, pencahayaan, material, resolusi, dan **anggaran segitiga yang sebanding**. Pembuat studi menolak build bila jumlah segitiga antar-metode pada satu LOD berbeda lebih dari toleransinya. Gunakan kode metode acak yang tidak mengungkap nama algoritma, acak urutan tampilan, dan simpan pemetaan kode metode terpisah dari materi peserta. Jangan mencantumkan label “metode usulan” pada materi penilaian.

Skala 1–5: `1 = sangat berbeda/rusak`, `2 = cukup berbeda`, `3 = sedang`, `4 = mirip`, `5 = sangat mirip`. Kesetiaan bentuk, silhouette, detail penting, dan kualitas keseluruhan dinilai terpisah terhadap sumber. Kolom komentar bersifat opsional. Rating tidak diisi otomatis; semua rating harus berasal dari peserta nyata.

## 6. Analisis dan batas kesimpulan

Hasil dipasangkan per aset dan rasio, dengan `Δ = PERCEPTUAL − baseline` terhadap QEM dan DECIMATE. Analisis (`scripts/analyze_results.py`) melaporkan hal berikut:

- Wilcoxon signed-rank eksak dua sisi (SciPy eksak, atau sign-flip eksak bila ada peringkat kembar).
- Estimasi pergeseran Hodges–Lehmann dengan CI eksak Walsh. Tingkat tercapainya 96,1% pada n = 9 dan 95,3% pada n = 7; CI tidak dilaporkan untuk n < 5.
- Korelasi rank-biserial.
- Menang/seri/kalah dengan toleransi seri 0,01 × |baseline| untuk jarak dan waktu, serta 0,0005 untuk IoU.
- Koreksi Holm atas keluarga yang dideklarasikan sebelum data dilihat.

Nilai p eksak terkecil yang mungkin adalah 2/2⁹ ≈ 0,0039 pada n = 9 dan 2/2⁷ ≈ 0,0156 pada n = 7. H2 karena itu hanya dapat lolos bila arahnya konsisten pada hampir semua aset. Per kategori (n = 3), p terkecil 0,25, sehingga analisis itu hanya deskriptif. Kolom `boot_ci_*` (bootstrap persentil) bersifat nominal dan terlalu sempit pada n sekecil ini, sehingga tidak boleh dikutip sebagai CI 95%. Sampel titik, view, dan pengulangan waktu bukan aset independen.

Semua analisis selain H1 dan H2 bersifat eksploratori: rasio 0,5 dan 0,2, perbandingan dengan DECIMATE, Chamfer dan Hausdorff global, non-ROI, IoU 6 sumbu, waktu, ablation, sapuan λ, run W1, dan eksperimen mask. Sapuan λ dan ablation pada split evaluasi tidak boleh dipakai untuk memilih ulang λ atau bobot. Penalaan ulang setelah melihat data evaluasi harus dilaporkan sebagai analisis eksploratori baru.

Jarak permukaan kecil belum tentu menunjukkan kualitas visual tinggi, dan IoU tinggi belum tentu mempertahankan wajah atau tekstur. Kanal silhouette dan metrik IoU sama-sama mengukur garis luar ortografik, sehingga kenaikan IoU sebagian merupakan konsekuensi langsung dari sasaran metode. Interpretasikan metrik bersama gambar, penilaian manusia, jumlah segitiga aktual, dan kegagalan. Hasil yang tidak mendukung hipotesis tetap merupakan hasil penelitian yang sah.

Prototipe mengolah snapshot mesh statis. Transfer UV/material bersifat pendekatan; seam, distorsi tekstur, normal khusus, dan atribut lain memerlukan pemeriksaan visual. Rig, skinning, shape key, dan animasi tidak dinilai atau dijamin dipertahankan. Ini juga bukan sistem pemilihan LOD runtime berdasarkan jarak kamera dan tidak menjamin bebas popping antar-LOD.

Implementasi Python cocok untuk prototipe yang dapat diperiksa, tetapi mesh besar memakan waktu dan memori. Waktu yang terukur (bagian 7) hanya berlaku untuk satu laptop pengembangan dan aset ≤ 100.000 segitiga. Klaim performa skala besar memerlukan pengukuran tersendiri.

## 7. Hasil

Rincian lengkap ada di [results/evaluation/analysis/report.md](../results/evaluation/analysis/report.md), dengan tabel di `analysis/tables/` (`paired_main`, `paired_by_category`, `per_asset`, `ablation`, `lambda_sweep`, `timing`, dan `all_quality_rows`, masing-masing `.md` + `.csv`). Gambar di `analysis/figures/`:

- `quality_vs_triangles_evaluation.png`: kualitas terhadap jumlah segitiga aktual.
- `lambda_pareto_evaluation.png`: sapuan λ, eksploratori tanpa penanda Pareto.
- `timing_vs_source_triangles.png`: waktu terhadap ukuran sumber.

Eksperimen mask: [results/evaluation/mask_assisted/summary.md](../results/evaluation/mask_assisted/summary.md) (data mentah `raw.jsonl`). LOD metode utama tersimpan di `results/evaluation/lods_all/<aset>.blend` sebagai `REF__<aset>` dan `<METODE>__L<level>__<aset>`. Laporan dibuat dari `quality_A`, `quality_B`, `quality_C`, dan `timing` dengan keluarga Holm sesuai praregistrasi:

```bash
python scripts/analyze_results.py results/evaluation/quality_A results/evaluation/quality_B \
    results/evaluation/quality_C results/evaluation/timing --out results/evaluation/analysis \
    --holm-metrics iou_multiview,roi_chamfer_sym_n --holm-baselines QEM --holm-ratios 0.05
```

Ringkasan (9 aset evaluasi, Blender 5.1.2, satu laptop i7-12700H):

| Uji primer (rasio 5 %, vs QEM) | n | HL Δ [CI eksak] | Menang/seri/kalah | p | p Holm | Keputusan |
| --- | ---: | --- | --- | ---: | ---: | --- |
| H1: IoU silhouette 13 arah | 9 | +0,00036 [+0,00018; +0,00093] | 3/6/0 | 0,0039 | 0,0078 | Didukung |
| H2: Chamfer ROI simetris | 7 | +1,6·10⁻⁵ [−2,5·10⁻⁵; +8,6·10⁻⁵] | 3/0/4 | 0,469 | 0,469 | Tidak didukung |

- **H1.** PERCEPTUAL lebih tinggi pada kesembilan aset (rank-biserial +1), tetapi efeknya kecil secara absolut. IoU 13 arah QEM pada 5% sudah 0,990–0,999, dan selisihnya 0,00007–0,00156. Galat `1 − IoU` turun 1,1–31,9% per aset (median 12,7%), dan enam selisih lebih kecil dari toleransi seri. Terhadap DECIMATE, selisihnya tidak jelas (p = 0,129; 2/6/1).
- **H2.** Tidak didukung, sesuai prediksi praregistrasi. Arahnya sedikit merugikan: median Chamfer ROI +6,6% terhadap QEM. Tiga aset membaik (beetle −15,0%, horse −11,6%, bunny −2,3%) dan empat memburuk (ogre +6,6%, max_planck +9,5%, igea +18,5%, armadillo +27,2%).
- **Biaya, eksploratori.** Hausdorff tersampel lebih buruk daripada QEM dengan median +46,1% / +62,6% / +30,5% pada rasio 50 / 20 / 5% (menang/seri/kalah 1/0/8, 0/0/9, 2/0/7). Hausdorff dengan vertex pada 5% tidak berbeda jelas (4/1/4, p = 0,73). Chamfer global naik dengan median +4,6% pada 50% dan 20% (pada 20%, 0/1/8, p = 0,0078) dan +3,0% pada 5% (p = 0,20).
- **Waktu.** PERCEPTUAL sekitar 4,2 s lebih lama per LOD daripada QEM (Hodges–Lehmann; 9/9 aset). Median satu set tiga LOD pada aset 97–100 ribu segitiga adalah 2,1–2,7 s untuk DECIMATE, 13,0–14,5 s untuk QEM, dan 17,3–20,0 s untuk PERCEPTUAL.
- **Ablation dan λ.** Aturan anchor jauh lebih buruk (bagian 3.4). Rata-rata IoU 13 arah saliency saja lebih rendah daripada silhouette saja pada semua rasio, walau sebagian besar selisih per aset masih dalam toleransi seri. Dalam sapuan, λ yang lebih besar cenderung menaikkan IoU sekaligus Chamfer ROI.
- **Mask manual** (sekunder, sirkular). Chamfer ROI simetris turun 10,8–23,3% terhadap QEM pada kesembilan kombinasi aset × rasio, sedangkan tanpa mask naik 6,6–28,6%. Biayanya ada di luar ROI: Chamfer non-ROI naik 1,9–55,2% dan Chamfer global naik 0,6–32,8%, terburuk pada ogre 50%.
- **Studi pengguna.** Paket stimulus `study/utama/` sudah dibangun, tetapi belum ada rating. H3 belum dapat diuji.

Kesimpulan yang dapat dipertahankan dari data ini: dengan konfigurasi hasil penalaan, kuadrik berbobot silhouette memberi kenaikan IoU silhouette yang kecil tetapi konsisten dibanding QEM tanpa bobot. Kenaikan itu dibayar dengan Hausdorff yang lebih besar dan waktu tambahan, dan tidak memperbaiki area penting seperti wajah. Perlindungan wajah memerlukan mask manual, yang bekerja tetapi memindahkan galat ke wilayah lain. Data ini tidak mendukung klaim kesesuaian dengan persepsi manusia sebelum studi pengguna selesai.

## 8. Berkas dan lisensi

Kode proyek menggunakan `GPL-3.0-or-later`; teks lisensi tersedia dalam [LICENSE](../LICENSE). Lisensi kode tidak otomatis memberi hak atas aset 3D milik pihak lain. Dataset dan gambar hasil perlu menyertakan atribusi dan lisensi aset masing-masing, sesuai [dataset/README.md](../dataset/README.md). Aset eksternal, LOD hasil, dan data peserta tidak di-commit. Berkas studi dan rancangan ini tidak memuat data eksperimen atau data peserta yang dibuat-buat.

Sumber primer yang dirujuk diverifikasi pada 21 September 2026. Simpan versi/pin sumber Blender ketika menulis laporan akhir, karena tautan branch `main` dan manual pengembangan dapat berubah.
