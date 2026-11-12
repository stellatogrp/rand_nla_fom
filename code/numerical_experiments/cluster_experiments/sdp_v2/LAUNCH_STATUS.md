# sdp_v2 launch status

## 2026-11-12: submitted

Requested by the user: try the one-sided, safeguarded projection variant
(`README.md` has the design). The user chose a pilot of task 0 followed by
tasks 1 to 11 in one array, two at a time.

Before submission: Ruff and all 432 tests passed locally (23 of them in
`tests/test_sdp.py`, among them that the methods of `sdp_v1` produce the same
iterates as the archived `releases/sdp_r1`); studies up to max-cut at
`n = 800` ran locally; `sync up --config slurm/sdp_v2.env --dry-run` listed
only `sdp/` (5 changed files), `sdp_v1/LAUNCH_STATUS.md`, `sdp_v2/`, and
`slurm/sdp_v2*.env`, and the upload transferred those 10 files. `studies_v3`
was still running; these jobs exclude its two nodes.

| Job | What | Array | Memory, time | Placement |
| --- | --- | --- | --- | --- |
| 17379819 | pilot: `planted_n40` | `0` | 4 GB, 0:30 | `--exclude=della-h12n14,della-h12n15` |
| 17379858 | the other eleven studies | `1-11%2` | 4 GB, 5:00 | `--exclude=della-h12n14,della-h12n15` |

The pilot completed in 20 s on `della-h12n18` (exit 0, 51 MB reported, empty
stderr; all 57 runs converged and passed every check of `sdp.summarize`), so
the remaining tasks were submitted unchanged.
