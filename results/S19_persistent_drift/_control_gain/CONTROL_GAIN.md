# Control gain: S19_persistent_drift

Seeds [20260925, 20260926, 20260927]; scored from epoch 100.

| | IF | i1 | i2 | i3 | i4 | i5 | i6 |
|---|---|---|---|---|---|---|---|
| FLOOR: provisioned (all-reject) | 0.7094 | 0.90 | 0.93 | 0.71 | 0.56 | 0.65 | 0.51 |
| CEILING: demand-tracking | 0.8513 | 0.83 | 0.86 | 0.94 | 0.89 | 0.76 | 0.83 |

- ceiling 0.8513 (want >= 0.85) -> OK
- gap +0.1418 (want >= 0.2) -> TOO SMALL: doing nothing is nearly as good as ideal control