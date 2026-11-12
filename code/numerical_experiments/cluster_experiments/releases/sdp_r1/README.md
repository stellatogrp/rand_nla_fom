# sdp_r1: the `sdp` package of sdp_v1

A byte-for-byte copy, taken on 2026-11-12, of `../../sdp/*.py` as run by the
`sdp_v1` campaign: the hash of these sources (file names and contents, in
name order, as `sdp.run.source_hash` computes it) is
`28af313fa57510c32db68391d6f477c4d1c6ab6d07e4f8ded74c44654dd6b695`, the
`source_sha256` of every `settings.json` of `results/sdp_v1`. `SHA256SUMS`
lists the files.

The working `sdp` package then gained the one-sided projection
(`projection="one_sided"`, the guard, and the Cholesky certificate). The
methods of `sdp_v1` are unchanged in it: `tests/test_sdp.py` checks that this
copy and the working package produce the same iterates, bit for bit. To rerun
`sdp_v1` exactly, run it with this copy in place of `sdp/`.
