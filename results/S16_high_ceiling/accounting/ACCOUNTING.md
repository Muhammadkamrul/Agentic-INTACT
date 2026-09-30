# Accounting: allocation cost and overhead (measured, not optimised)

Scenario `S16_high_ceiling`. Verified replays: 8.
No method was designed to minimise any of these quantities, and none was changed. **A lower bill is not automatically better**: stranding capacity lowers the lease while failing intents. Read cost next to fulfilment.

## Billing proxies (per run, scored epochs; PRB-s = PRB-seconds)

| method | IF | reserved PRB-s | used PRB-s | idle leased PRB-s | unsold pool PRB-s | delivered Gbit | PRB-s per fulfilled intent-epoch |
|---|---|---|---|---|---|---|---|
| All-reject | 0.751 | 6784 | 3884 | 2900 | 0 | 2.5 | 3.78 |
| B0 all-admit | 0.605 | 4225 | 3603 | 635 | 2559 | 2.4 | 2.93 |

### Per-tenant lease (reserved PRB-s)

| method | T1 | T2 | T3 |
|---|---|---|---|
| All-reject | 2944 | 1920 | 1920 |
| B0 all-admit | 1085 | 1638 | 1502 |

## Overhead proxies (per epoch unless stated)

| method | E2 control msgs | reconfiguration loss (% of epoch capacity) | candidates scored | decision p95 (ms) | twin calls / run | twin s / run |
|---|---|---|---|---|---|---|
| All-reject | 0.00 | 0.00 | 1.0 | 0.13 | 0 | 0 |
| B0 all-admit | 2.90 | 2.88 | 1.0 | 0.33 | 0 | 0 |

## How to read this

- **Reserved PRB-s** is what a tenant would pay under hard slicing. A method with a low value may simply be stranding capacity; check **unsold pool PRB-s** and IF beside it.
- **PRB-s per fulfilled intent-epoch** is the cost-effectiveness figure: capacity leased per unit of fulfilment delivered.
- **E2 control messages** equal executed writes (one control request each); **reconfiguration loss** is capacity the simulator actually removed during reconfiguration transients.
- Twin work runs in the slow loop and is reported separately from the near-RT decision latency.