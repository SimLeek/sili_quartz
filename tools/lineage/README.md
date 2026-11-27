# Lineage orchestration (track A evolution)

Copies of the scripts that run the track A lineage on the sandbox (they live in
`~/runs/` there and assume its paths). See
`docs/energy-controller/track-a-required-pieces.md`.

| script | does |
|---|---|
| `eval_node.sh TAG` | scorecard against Root_g (10-minute windows), repertoire probe and active fan-in on the node's last checkpoint |
| `decide_node.py CHILD PARENT` | advance or reject a child against its parent |
| `select_sibling.py TAG...` | winner and runner-up among siblings (health gate, then input coherence); reruns `TAG_s1` are averaged with `TAG` |
| `plan_combos.py TAG...` | combination siblings from the healthy single drives |
| `noise_floor.py` | spread across the untouched runs (play level) and the probe's sampling sd (probe level) |
| `margin_check.py W R` | CLEAR if the winner's lead exceeds the noise floor, else RERUN with a second seed |
| `sweep_report.py W` | the whole spread over learning rate and dose at 1/4x, 1x, 4x: every point, best, worst, range, best-at-edge, whether metrics agree |
| `queue24e.sh` | the drive generation: noise floor, singles, combinations, selection (with reruns), sweep, children |
| `flags_for.py TAG` | trainer and actor flags of a drive-generation node, from its tag |
| `randinit_report.py CTRL TAG...` | random-init checks: each trained node against the same random start with no training |
| `queue25.sh` | random-init checks of the generation's winner and runner-up (re-initialized last chunk; scrambled keys) |
