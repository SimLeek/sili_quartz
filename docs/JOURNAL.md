# Journal: properties that turned out not to hold

These were ledger rows in `PROPERTIES.md`. Each one was tested and does not
hold ("not true"), so they are no longer properties the models must have.
Properties that are only not yet shown ("not proven true") stay in the ledger
as open. The
numbers and tests stay here as the record. Row numbers are the old ones, so
older results docs still resolve.

## Row 27: B (RMT) learns beyond-window recall through passed tokens alone

**FAILS, strict-xfail. Measured 0.33±0.01 for m=4 and m=16, with and without a 0→4→8→16 gap curriculum (4 runs). Per-pair accuracy uniform across write order (0.31–0.36): it carries the SET of written values (1/3 ≈ guessing among them) but never binds key→value.**

The strict-xfail test `test_family_learns_long_gap_recall[rmt]` stays in
`tests/test_mac.py`: if RMT ever starts binding, it will flip and say so.

## Row 31: poly2 + omega trains at scale

**FAILS** at full size and at `poly_rank=16`. Alone each trains: omega 3/3 seeds (2–4k 51.8); full poly2 2/2 (58.1, 48.2; 13 GB); poly2 r16 1/2 (59.5, seed 2 collapses; 1.06 GB). **poly2 dropped** (user, 2026-11-17).

## Row 32: training escapes the start for every seed

**FAILS** in 3 of 4 arms. Titans + omega trains 7/7. The failing seed moves with the config (titans/combined s1, combined + omega s5) and with the GPU: combined + omega s2 stalled on ai0 but trained at 4.5k on the 2060. No test yet.

This is still wanted as a fixture: a hard start under Adam that the sili
optimizer may recover (TODO 1a.4). It is not a property the torch reference has.
