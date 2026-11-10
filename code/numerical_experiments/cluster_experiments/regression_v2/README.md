# Next regression pilot batch

Four pilots use the new, sealed regression_v9 engine and the unchanged native
labels/features and objectives. Results are separate from every earlier run.
All runs retain tolerance 1e-6, warm inner initialization, stride 1, one BLAS
thread, independent loading/calibration/setup, and charged true-certificate work.

| Task | Dataset | Method | L/c or gamma*L | Sigma | Purpose |
| --- | --- | --- | --- | --- | --- |
| 0 | realsim | LASSO Nystrom-PCG, rank 128 | 1e4 | 0.2 | Rerun the passing-final-certificate case with corrected stopping |
| 1 | realsim | Logistic vanilla GD | 100 | 0.99 | Reduce inner oversolving through looser relative acceptance |
| 2 | smallNORB | LASSO Nystrom-PCG, rank 128 | 1e5 | 0.2 | Test an intermediate penalty between the previous pilot values |
| 3 | smallNORB | Logistic vanilla GD | 100 | 0.99 | Compare with the previous sigma 0.9 pilot |

The extra graph gate is removed from common-certificate termination. The true
certificate is checked initially, at outer acceptance, every ten inner updates,
and at stopping-budget boundaries. This is an empirical pilot batch; no new
model-optimal parameters or convergence times are claimed before results exist.

Every run has a 7,200-second solve budget, 3,600-second per-inner budget,
100,000 total inner updates, 10,000 per outer solve, and 10,000 outer solves.
The worker has a 56 GB address-space cap and an 8,100-second hard wall timeout
(7,200 solve plus 900 setup allowance). Each Slurm task requests one CPU,
64 GB, 2:30:00, and no GPU. At most two tasks run concurrently. Prepared
sources, launchers, inputs, and the numerical archive are checked by content
hash before each worker; existing frozen releases and results are preserved.
