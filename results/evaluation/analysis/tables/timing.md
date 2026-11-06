# Timing, median over non-warm-up repetitions

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
