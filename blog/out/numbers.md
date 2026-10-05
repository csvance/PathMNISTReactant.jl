# Test-set numbers

- macro accuracy 95.93%, accuracy 96.77%, ECE 1.68%
- errors: 232 of 7180
- reliability: 6785 tiles in the top bin; largest lower bin holds 108
- test loss (cross-entropy): 0.132
- errors above 99% confidence: 51
- of the 12 most confident errors, stroma/smooth muscle either way: 12

## Per-class recall

| Class | Recall | Tiles |
|---|---:|---:|
| adipose | 99.10% | 1338 |
| background | 100.00% | 847 |
| debris | 99.12% | 339 |
| lymphocytes | 100.00% | 634 |
| mucus | 94.20% | 1035 |
| smooth muscle | 89.36% | 592 |
| normal mucosa | 98.79% | 741 |
| stroma | 84.32% | 421 |
| tumor epithelium | 98.46% | 1233 |

## Largest confusions

| True | Predicted | Count |
|---|---|---:|
| smooth muscle | stroma | 51 |
| stroma | smooth muscle | 50 |
| mucus | smooth muscle | 31 |
| mucus | adipose | 19 |
| tumor epithelium | normal mucosa | 13 |
| adipose | smooth muscle | 11 |
| stroma | debris | 10 |
| smooth muscle | mucus | 10 |

## Trajectories

- decision time t (final class stays above 0.5): 10% 0.18, 25% 0.18, median 0.20, 75% 0.22, 90% 0.26
