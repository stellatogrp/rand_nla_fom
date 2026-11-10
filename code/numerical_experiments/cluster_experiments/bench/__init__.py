"""One-node timing benchmarks of the original-format studies (``studies``).

A campaign runs every configuration as its own Slurm task, so its wall-clock
times come from whichever shared node and load each task met (see
``studies_v2/LAUNCH_STATUS.md``).  :mod:`bench.one_node` re-times a chosen
set of configurations of every inner solver in one process on one node,
interleaved instance by instance, from the campaign's stored setups, so that
the solvers of a problem are compared under the same conditions.
"""
