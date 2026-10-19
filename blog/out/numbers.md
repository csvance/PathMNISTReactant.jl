# Test-set numbers

- macro accuracy 95.77%, accuracy 96.81%, ECE 1.46%
- errors: 229 of 7180
- reliability: 6716 tiles in the top bin; largest lower bin holds 129
- test loss (cross-entropy): 0.111
- errors above 99% confidence: 44
- of the 12 most confident errors, stroma/smooth muscle either way: 11

## Per-class recall

| Class | Recall | Tiles |
|---|---:|---:|
| adipose | 98.51% | 1338 |
| background | 100.00% | 847 |
| debris | 97.05% | 339 |
| lymphocytes | 100.00% | 634 |
| mucus | 95.56% | 1035 |
| smooth muscle | 92.23% | 592 |
| normal mucosa | 98.25% | 741 |
| stroma | 81.71% | 421 |
| tumor epithelium | 98.62% | 1233 |

## Largest confusions

| True | Predicted | Count |
|---|---|---:|
| stroma | smooth muscle | 53 |
| smooth muscle | stroma | 23 |
| mucus | smooth muscle | 20 |
| stroma | debris | 17 |
| adipose | smooth muscle | 17 |
| smooth muscle | debris | 16 |
| mucus | adipose | 15 |
| tumor epithelium | normal mucosa | 12 |

## Trajectories

- decision time t (final class stays above 0.5): 10% 0.18, 25% 0.18, median 0.20, 75% 0.22, 90% 0.28
