# Selected model: fixed TabPFN ownership with short-control preservation

The selected full-match candidate is `data/candidate_tabpfn_short_control_v1/candidate.csv`.
It contains 199,319 rows and has SHA256 `4632cfe2cb71ffb15e76784b257175168e1e634754a48a612edfd3cdf6baa58e`.

Prespecified seed42 retains the confirmed public main SpellF1 **0.7771428571**, compared with0.7411764706 for the previous motion model. Purged and longer-block SpellF1 are0.7314285714 and0.6810810811. All nine matched comparisons across seeds42/43/44 passed; seed44's higher main result did not change the prespecified production seed. Independent reconstruction and complete five-main-fold replay under the revised allocation reproduced probabilities and labels exactly.

The learner,2352-row full paired context, eight views,float32 precision,512-row batches and frozen short-control correction are unchanged. The memory fix assigns5GiB to MPS and reserves1GiB for the host within the same6GiB combined limit. Full-match inference completed40 fresh context fits over158857 official-half frames. All worker integrity/resource checks passed, both separately loaded full CSV exports match exactly, and both pass the official validator. D/X predictions remain exact to the previous full candidate.

The package manifest, full probability evidence and final audit are under `data/candidate_tabpfn_short_control_v1/`; detailed production evidence is under `data/control_tabpfn_full_build_v3`. Two CSV exports were rebuilt from the completed full probabilities; the entire full-match native query inference was not performed twice. The public validation evidence reuses7500 same-match labels; private full-match accuracy remains unknown.

The previous full candidate and all17 protected artifacts remain unchanged under `data/candidate_motion_dead_trim/`. The exact preceding selected-model document is preserved in [SELECTED_MODEL_BEFORE_TABPFN_20261102.md](SELECTED_MODEL_BEFORE_TABPFN_20261102.md).
