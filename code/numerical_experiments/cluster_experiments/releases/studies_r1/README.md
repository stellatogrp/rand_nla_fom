# studies_r1: the `studies` package of studies_v1, studies_v2, studies_v3, and bench_v1

A byte-for-byte copy, taken on 2026-11-11, of `../../studies/*.py` as sealed by
`studies_v1`, `studies_v2`, `studies_v3`, and `bench_v1` (`SHA256SUMS` equals
the `studies/*.py` entries of all four seals). The working `studies` package
then changed its calibration (time every step of solvers whose step cost does
not depend on the step, 30 repetitions by default, calibrate on the first
instance's own setup), so those campaigns no longer verify against the working
tree; `studies.campaign.verify(manifest, release="studies_r1")` verifies them
against this copy instead.

To rerun or re-collect one of them, run it with this copy in place of
`studies/`. On Della, `studies_v3` was launched with this code: do not
`sync up` the changed `studies/` until its jobs have finished.
