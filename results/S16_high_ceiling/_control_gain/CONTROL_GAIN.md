# Control gain: S16_high_ceiling

Seeds [20260925, 20260926, 20260927]; scored from epoch 100.

| | IF | i1 | i2 | i3 | i4 | i5 | i6 |
|---|---|---|---|---|---|---|---|
| FLOOR: provisioned (all-reject) | 0.7236 | 0.96 | 0.96 | 0.64 | 0.48 | 0.71 | 0.60 |
| CEILING: demand-tracking | 0.9146 | 0.95 | 0.96 | 0.92 | 0.85 | 0.85 | 0.96 |

- ceiling 0.9146 (want >= 0.85) -> OK
- gap +0.1910 (want >= 0.2) -> TOO SMALL: doing nothing is nearly as good as ideal control