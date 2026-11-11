# R63 TabPFN validation initialization: failed full-context gate

R63 is **FAILED**, not eligible for native processing. The independently reviewed
one-time initialization resolved the traced lazy-validation metadata issue, but
the full 3,000-row context hit the internal MPS allocator limit on its first
512-row query. No full rebuild, native feature processing, possession score or
candidate promotion occurred. All preceding failures remain preserved.

| Stage | Exit | Result | Wall seconds | Supervised combined peak GiB |
|---|---:|---|---:|---:|
| Moderate first, 64 × 166 | 0 | pass | 69.884 | 1.855 |
| Moderate fresh rebuild | 0 | pass | 66.357 | 1.869 |
| Full first, 3,000 × 166 | 1 | MPS allocation failure | 21.102 | 4.151 |
| Full fresh rebuild | — | not reached | — | — |

The authoritative records are under
`data/control_tabpfn_mps_validation_init_v1/runtime_assessment/`. Execution
session83926 ended with exit1; process inspection confirmed no R63 worker remains.
Root independently compared all14 moderate NumPy artifacts and the complete
recipe/checkpoint/preprocessing-fingerprint JSON files across fresh processes:
every byte matches. Both initialization records contain exactly eight completed
author-method calls; all existing fitted/configuration attribute hashes remain
unchanged. Complete raw fingerprints, checkpoint tensors, canonical repeats and
the official exclusive MPS attention-wrapper checks pass in those two stages.
Noncanonical short-prefix predictions differ by at most1.7881393e-7; that
predeclared diagnostic does not select a query schedule or waive exact canonical
reconstruction.

Full-context fitting and initial checkpoint verification finish. The first
prediction then raises `TabPFNMPSOutOfMemoryError`: the internal allocator reports
2.67GiB allocated plus a624MiB request against3GiB allowed. The supervisor records
peak full driver memory3.328GiB and owned RSS.823GiB, with peak conservative
combined accounting4.151GiB. Driver accounting includes allocations outside the
allocator's tensor limit, so the3GiB setting must not be described as a strict
driver cap. The enforced combined6GiB ceiling was not exceeded.

This motivates a **separate prospective R66 resource-allocation experiment**:
4GiB internal MPS allowance and2GiB planning reserve for CPU, with the same enforced
combined owned-RSS-plus-full-driver6GiB ceiling. Its fraction is fixed before any
new model call. Model, math, checkpoint, eight views, float32 precision, 512-row
canonical schedule, all four stages, full raw fingerprints, exact reconstruction,
300-second operation limits and900-second worker limits remain unchanged. R66
must pass independently; R63 is neither retried nor relabelled a success.

Source SHA256: `59823fca874ead58dbd1b8684b17647d2dea61cff53523466b70dde05657494f`.
Plan SHA256: `5c8ecead817a74645218b1ea6455d481d4ec0da9290f6fd0a7d92c86c078e6fe`.
The root preflight and [independent review](R63_INDEPENDENT_REVIEW.md) preceded
freeze and execution. Built with PriorLabs-TabPFN; original author notices and
checkpoint license remain retained.
