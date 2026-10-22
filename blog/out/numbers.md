# Test-set numbers

- macro accuracy 95.04%, accuracy 96.09%, ECE 2.17%
- errors: 281 of 7180
- reliability: 6751 tiles in the top bin; largest lower bin holds 119
- test loss (cross-entropy): 0.149
- errors above 99% confidence: 61
- of the 12 most confident errors, stroma/smooth muscle either way: 10

## Per-class recall

| Class | Recall | Tiles |
|---|---:|---:|
| adipose | 98.13% | 1338 |
| background | 100.00% | 847 |
| debris | 99.71% | 339 |
| lymphocytes | 99.84% | 634 |
| mucus | 94.88% | 1035 |
| smooth muscle | 83.11% | 592 |
| normal mucosa | 97.84% | 741 |
| stroma | 82.90% | 421 |
| tumor epithelium | 98.95% | 1233 |

## Largest confusions

| True | Predicted | Count |
|---|---|---:|
| smooth muscle | stroma | 68 |
| stroma | smooth muscle | 48 |
| smooth muscle | debris | 31 |
| mucus | smooth muscle | 24 |
| adipose | smooth muscle | 20 |
| mucus | adipose | 16 |
| stroma | tumor epithelium | 11 |
| stroma | debris | 11 |

## Trajectories

- decision time t (final class stays above 0.5): 10% 0.18, 25% 0.20, median 0.20, 75% 0.22, 90% 0.28
