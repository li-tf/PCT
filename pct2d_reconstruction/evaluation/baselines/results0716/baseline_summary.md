# results0716 frozen baseline summary

- Status: PASS
- Generated: 2026-07-17T09:31:06+08:00
- Truth: 200 MeV reference RSP
- Split: `splitmix64_v1(RunID, filtered_row_index, 20260713) % 10 == 0`
- Training pairs: 219,790,828
- Validation pairs: 24,426,971 (10.0021%)
- Validation angles: 720

| Reconstruction | Water mean | Water std | Phantom RSP RMSE | Al platform recovery | CNR | Edge 10-90 (mm) | Validation WEPL RMSE (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-Hann DDB-FDK | 1.013718 | 0.009720 | 0.045075 | 98.8901% | 106.92 | 1.1367 | 2.6509 |
| GPU OS-SART + Huber-TV, epoch 3 | 1.014063 | 0.002453 | 0.041956 | 98.7113% | 399.53 | 1.1179 | 2.6210 |

MTF and path-error fields are intentionally unavailable for this experiment. The WEPL values above are computed on the fixed validation split and are distinct from the online pre-update training residuals saved by reconstruction.

Validation forward projection used `NVIDIA GeForce RTX 4060 Laptop GPU` and took 490.9 s.
