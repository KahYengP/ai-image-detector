| Condition | Severity | AUC | N |
| --- | --- | --- | --- |
| Clean | - | 0.5469 | 16 |
| JPEG q90 | 90 | 0.3438 | 16 |
| JPEG q70 | 70 | 0.2031 | 16 |
| JPEG q50 | 50 | 0.5156 | 16 |
| JPEG q30 | 30 | 0.6719 | 16 |
| Blur sigma=0.5 | 0.5 | 0.5469 | 16 |
| Blur sigma=1.0 | 1.0 | 0.5625 | 16 |
| Blur sigma=2.0 | 2.0 | 0.4531 | 16 |
| Resize 0.5x | 0.5 | 0.5469 | 16 |
| Resize 0.25x | 0.25 | 0.5625 | 16 |
| Noise sigma=0.02 | 0.02 | 0.4531 | 16 |
| Noise sigma=0.05 | 0.05 | 0.4219 | 16 |
| Noise sigma=0.10 | 0.1 | 0.3906 | 16 |
| Color jitter +/-20% | 0.2 | 0.5469 | 16 |
| Crop 80% | 0.8 | 0.7188 | 16 |

AUC_clean = 0.546875
AUC_robust = 0.4955357142857143  (mean of transformed conditions)
Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust = 0.5212053571428572
