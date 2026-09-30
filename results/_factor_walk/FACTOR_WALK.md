# Factor walk

Base `S16_high_ceiling` (cell-level regime wins), donor `S8_dedicated_contest` (per-tenant regime wins). Frozen methods, development seeds [20260925]. Margin for a clear ordering: 0.02.

| variant | all-reject | B3 | INTACT-RA (per-tenant) | INTACT-RA (cell) | per-tenant − cell | ordering | best controller − all-reject |
|---|---|---|---|---|---|---|---|
| BASE | 0.709 | 0.780 | 0.673 | 0.778 | -0.105 | CELL WINS | +0.070 |
| shared_knobs | 0.709 | 0.765 | 0.573 | 0.764 | -0.190 | CELL WINS | +0.055 |
| loads | 0.682 | 0.635 | 0.528 | 0.634 | -0.106 | CELL WINS | -0.047 |
| migration_out | 0.719 | 0.578 | 0.576 | 0.578 | -0.002 | no clear order | -0.141 |
| shared_knobs+host_intent | 0.751 | 0.655 | 0.618 | 0.739 | -0.121 | CELL WINS | -0.012 |
| percentile | 0.491 | 0.421 | 0.424 | 0.420 | +0.003 | no clear order | -0.067 |
| no_burnin | 0.696 | 0.709 | 0.665 | 0.727 | -0.062 | CELL WINS | +0.031 |
| percentile+no_burnin | 0.490 | 0.414 | 0.413 | 0.414 | -0.000 | no clear order | -0.076 |

## Interpretation

No variant here shows the per-tenant regime winning while control beats doing nothing. Try combinations of factors (join with '+'), or factors not yet transplanted.