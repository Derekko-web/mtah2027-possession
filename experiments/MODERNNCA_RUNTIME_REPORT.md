# R64 ModernNCA runtime report

**PASS: all four prospectively frozen invented-data stages completed and all retained artifact/hash and fresh-refit checks verify.** The gate establishes local technical feasibility at the measured sizes; it does not authorize native processing or establish possession accuracy.

| Stage | Fit seconds | Worker seconds | Peak supervised/worker GiB | Bound cached artifacts | Longest query seconds |
|---|---:|---:|---:|---:|---:|
| moderate/first | 3.593151 | 11.951896 | 0.690491 | 23 | 0.062440 |
| moderate/rebuild | 3.587369 | 11.921138 | 0.693115 | 23 | 0.062779 |
| full/first | 262.654838 | 275.676469 | 4.018600 | 23 | 0.391896 |
| full/rebuild | 261.421823 | 274.255467 | 3.992126 | 23 | 0.396670 |

The stage order is moderate64×166 first, moderate64×166 independent fresh200-epoch refit, full3000×166 first, full3000×166 independent fresh200-epoch refit. All workers exit0 with no abort. All 92 stage cache SHA-256 bindings, four summary hashes and eight stdout/stderr hashes verify against retained evidence. Logs are retained. The R64 frozen implementation/runtime/symlink closure and source contract independently verify without scientific imports.

Each stage retains the exact author PLR-lite default (77 frequencies, scale .04431360576139521, width34, metric128, zero post-encoder blocks), AdamW .01/.0002, batch1024, float64CPU2, exactly200 complete epochs and the final checkpoint. Extracted author candidate exclusion occurs before .5 neighbor sampling; batch self-diagonal exclusion and unsquared Euclidean retrieval are unchanged. Both invented contexts are paired to balance H0/A1; query sets contain 513/512 rows and use canonical512 last-row padding.

All required finite-loss/probability, exact symmetry/repeat, strict saved-state reload, frozen context/model state and resource assertions pass. For each size the fresh refit reproduces initial state fingerprints, final complete tensor-state fingerprints, recipe and optimizer bytes, every minibatch/epoch loss and original/swapped/symmetrized probabilities byte-for-byte. Torch archive container bytes are deliberately not used as the tensor equality criterion. First-reference stages correctly retain independent_refit_exact=false; only the two rebuild stages certify it.

Fit≤1800s, each query≤300s, worker≤2400s, peakRSS≤6GiB, CPU2 and serial scheduling remain unchanged. The largest supervised RSS is 4.018600GiB. No checkpoint/epoch, dtype, backend, batch, context cap, learning-rate or resource waiver was introduced after an outcome.

Frozen implementation SHA-256: `ecfc2ed97ebe9277c78995dcfb29543ba363ffc1b8ad66e84fda63c3e09f05d5`. Frozen plan SHA-256: `0b00303db7ad76a4181c242239c4663ccce69fc2ad9ad0afa4a7425d58ed6890`. Actual all-four-stage summary SHA-256: `f00bc2d9fc05456c7decd795fc60898266cec962bf4894196b7aa2ebc3dad47e`.

Prefix/order diagnostics remain diagnostic only and did not select a schedule or settings. Public17 possessions, actual native arrays, labels, model or score were not processed by R64. No private/manual/sealed data, external hosted inference, submission, commit or upload occurred. A separately reviewed/frozen R68 native plan is required before any native payload or model work; all confirmation/replay/full-public/double-CSV/validator/promotion gates remain outstanding.

This independent report reads frozen source/metadata, retained JSON and artifact bytes for hashing/exact replay only. It did not import a scientific package, load an array/model, rerun a fit or modify any R64 source/evidence.
