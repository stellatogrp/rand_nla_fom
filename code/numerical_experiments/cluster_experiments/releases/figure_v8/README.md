# Figure instrumentation v8

Derived from the sealed accounting_v7 numerical engine. Numerical updates,
convergence criteria, setup accounting, resource limits, and RNG streams are
preserved. Opt-in scalar inner traces record checked local graph residuals
and original-problem convergence diagnostics. Trace computation, allocation,
and CSV I/O remain charged to the independent worker wall time. Intermediate
CG/Nystrom diagnostics using recurrence gradients are explicitly identified;
terminal certificates continue to recompute the true gradient.

The theory comparison uses the verified deterministic GD/CG contraction
inequality for the inner graph residual. It does not claim an outer KKT rate
or a total stopping-work theorem from unrecorded exact fixed-point residuals.
