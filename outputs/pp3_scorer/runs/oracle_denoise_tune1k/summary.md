# Oracle denoising test (1000 samples)

Full-token accuracy 0.7080; baseline wrong on 292. Mean share of tokens with negative signed saliency: 0.495.

| drop | arm | mean dropped | accuracy | gained | lost | diff (pts) | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | oracle | 0.051 | 0.7340 | 47 | 21 | +2.60 | [+0.99, +4.21] | 0.002 |
| 0.05 | random | 0.051 | 0.7040 | 5 | 9 | -0.40 | [-1.13, +0.33] | 0.424 |
| 0.1 | oracle | 0.102 | 0.7420 | 58 | 24 | +3.40 | [+1.64, +5.16] | 0.000 |
| 0.1 | random | 0.102 | 0.7050 | 5 | 8 | -0.30 | [-1.01, +0.41] | 0.581 |
| 0.25 | oracle | 0.250 | 0.7400 | 63 | 31 | +3.20 | [+1.31, +5.09] | 0.001 |
| 0.25 | random | 0.250 | 0.7100 | 19 | 17 | +0.20 | [-0.98, +1.38] | 0.868 |

Oracle gains by GQA structural type (largest drop fraction):

| type | n | gained | lost |
| --- | --- | --- | --- |
| query | 510 | 34 | 15 |
| verify | 201 | 12 | 9 |
| logical | 132 | 8 | 6 |
| choose | 121 | 6 | 1 |
| compare | 36 | 3 | 0 |
