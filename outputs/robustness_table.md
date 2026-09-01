| Condition | Severity | AUC | N |
| --- | --- | --- | --- |
| Clean | - | 0.9606 | 43 |
| JPEG q90 | 90 | 0.9560 | 43 |
| JPEG q70 | 70 | 0.9514 | 43 |
| JPEG q50 | 50 | 0.9491 | 43 |
| JPEG q30 | 30 | 0.9144 | 43 |
| Blur sigma=0.5 | 0.5 | 0.9537 | 43 |
| Blur sigma=1.0 | 1.0 | 0.9653 | 43 |
| Blur sigma=2.0 | 2.0 | 0.9583 | 43 |
| Resize 0.5x | 0.5 | 0.9375 | 43 |
| Resize 0.25x | 0.25 | 0.9375 | 43 |
| Noise sigma=0.02 | 0.02 | 0.9514 | 43 |
| Noise sigma=0.05 | 0.05 | 0.9236 | 43 |
| Noise sigma=0.10 | 0.1 | 0.8588 | 43 |
| Color jitter +/-20% | 0.2 | 0.9630 | 43 |
| Crop 80% | 0.8 | 0.9514 | 43 |

AUC_clean = 0.9606481481481481
AUC_robust = 0.9408068783068784  (mean of transformed conditions)
Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust = 0.9507275132275133
